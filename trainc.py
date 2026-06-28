import os
import glob
import random
import argparse
import numpy as np
import pyarrow as pa  # 🌟 新增：处理 Arrow 文件
import io             # 🌟 新增：处理二进制流
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split # 🌟 新增：random_split 用于划分数据集
import torchvision.transforms.functional as TF

from transformers import AutoModel
from peft import LoraConfig, get_peft_model

# ==========================================
# 🌟 NPU 核心适配：导入昇腾支持
# ==========================================
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu # 协助自动路由部分底层 cuda 调用至 npu
except ImportError:
    print("⚠️ 未检测到 torch_npu 模块，请确保在昇腾 NPU 环境中运行！")

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 模型架构定义 (⚠️ 完全保持你的原样)
# ==========================================
class Norm(nn.Module):
    def __init__(self, mode='clip'):
        super().__init__()
        self.mode = mode
        if mode == 'clip':
            self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        else:  
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

class DINO(nn.Module):
    def __init__(self, dinov3_path, finetune=True):
        super(DINO, self).__init__()
        print(f"Loading Backbone from: {dinov3_path}")
        self.dino = AutoModel.from_pretrained(dinov3_path, weights_only=False)
        self.dino.requires_grad_(False)
        self.is_v3 = hasattr(self.dino, "layer") or "dinov3" in dinov3_path.lower()
        if finetune:
            self._apply_lora()

    def _apply_lora(self):
        target_modules = ["q_proj", "k_proj", "v_proj"] if self.is_v3 else ["query", "key", "value"]
        config = LoraConfig(r=8, lora_alpha=16, target_modules=target_modules)
        # 🌟 增量保存修复：直接使用 get_peft_model 包裹整个原生模型
        self.dino = get_peft_model(self.dino, config)
        self.dino.print_trainable_parameters()

    def forward(self, x):
        outputs = self.dino(pixel_values=x)
        last_hidden_state = outputs[0]
        feat_cls = last_hidden_state[:, 0]
        feat_tokens = last_hidden_state[:, 1:]
        return feat_tokens, feat_cls

class AdvancedPatchClassifier(nn.Module):
    def __init__(self, feat_dim=1280, hidden_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),  
            nn.GELU(),                 
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, feat_tokens, feat_cls):
        mean_pooled = feat_tokens.mean(dim=1)
        max_pooled, _ = feat_tokens.max(dim=1)
        combined = torch.cat([mean_pooled, max_pooled, feat_cls], dim=-1)
        return self.head(combined)

class Enhanced_Detector(nn.Module):
    def __init__(self, dino_path, feature_dim=1280, finetune=True):
        super().__init__()
        self.norm = Norm(mode='imagenet')
        self.backbone = DINO(dino_path, finetune=finetune) 
        self.detector = AdvancedPatchClassifier(feat_dim=feature_dim)

    def forward(self, x):
        x = self.norm(x)
        feat_tokens, feat_cls = self.backbone(x)
        logits = self.detector(feat_tokens, feat_cls)
        return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

# ==========================================
# 数据处理与加载 (🌟 仅修改图片加载逻辑)
# ==========================================
class DynamicScaleCenterCrop:
    def __init__(self, crop_size=224):
        self.crop_size = crop_size
    def __call__(self, img):
        short_edge = min(img.size)
        if short_edge < 256:
            img = TF.resize(img, 256, interpolation=TF.InterpolationMode.BICUBIC)
        elif short_edge > 512:
            img = TF.resize(img, 512, interpolation=TF.InterpolationMode.BICUBIC)
        # NPU 极度依赖静态形状，CenterCrop 保证了每次输出都是恒定的 224x224
        img = TF.center_crop(img, [self.crop_size, self.crop_size])
        return TF.to_tensor(img)

# 🌟 替换了原有的 BaselineDataset
class CAIDArrowDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.tables = []
        self.offsets = []
        self.total_length = 0
        
        arrow_paths = glob.glob(os.path.join(root_dir, '**', 'test.arrow'), recursive=True)
        if not arrow_paths:
            raise ValueError(f"在 {root_dir} 中未找到 test.arrow 文件！")

        print(f"📦 发现 {len(arrow_paths)} 个 Arrow 文件，正在自适应加载...")
        
        for path in tqdm(arrow_paths, desc="解析 Arrow"):
            try:
                with pa.OSFile(path, 'rb') as f:
                    try:
                        table = pa.ipc.open_file(f).read_all()
                    except pa.lib.ArrowInvalid:
                        f.seek(0)
                        table = pa.ipc.open_stream(f).read_all()
                        
                self.tables.append(table)
                self.offsets.append(self.total_length)
                self.total_length += table.num_rows
            except Exception as e:
                tqdm.write(f"\n⚠️ 警告: {path} 读取彻底失败。报错: {e}")

        print(f"✅ 成功加载完成！总共提取到图片: {self.total_length} 张。")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        table_idx = 0
        for i in range(1, len(self.offsets)):
            if idx < self.offsets[i]:
                break
            table_idx = i
            
        local_idx = idx - self.offsets[table_idx]
        table = self.tables[table_idx]

        # 读取二进制图片并翻转标签对齐逻辑
        img_bytes = table['image'][local_idx].as_buffer()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        label = 1 - table['label'][local_idx].as_py() 

        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)

# ==========================================
# 训练主循环 (NPU 优化版)
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/data/datasets/DDA-Training-Set/COCO-SD-2')
    parser.add_argument('--backbone_path', type=str, default='/data/models/facebook/dinov3-vith16plus-pretrain-lvd1689m')
    parser.add_argument('--save_dir', type=str, default='./weight')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    # 🌟 NPU 核心适配：指定 device 为 npu
    device = torch.device('npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu')
    print(f"🖥️ 当前使用的计算设备: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 🌟 修改点：使用 ArrowDataset 并在内部完成划分
    print("📦 正在加载并划分 Arrow 数据集...")
    transform = DynamicScaleCenterCrop(224)
    full_dataset = CAIDArrowDataset(root_dir=args.data_root, transform=transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 🌟 NPU 核心适配：drop_last=True 防止最后的非完整 batch 触发图重新编译
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)

    print("🚀 初始化优化模型 (DINOv3 LoRA + Advanced Classifier 头)...")
    model = Enhanced_Detector(dino_path=args.backbone_path, feature_dim=1280, finetune=True).to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = FocalLoss(gamma=2.0, reduction='mean')
    
    # 🌟 NPU 核心适配：混合精度 scaler 指定为 npu
    scaler = torch.amp.GradScaler('npu')

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 🌟 NPU 核心适配：混合精度 autocast 指定为 npu
            with torch.amp.autocast('npu'):
                logits = model(imgs)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        scheduler.step()

        # 验证评估
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                with torch.amp.autocast('npu'):
                    logits = model(imgs)
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(labels.numpy())
                
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"Epoch {epoch+1} 验证集准确率: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_dir = os.path.join(args.save_dir, "best_lora_model")
            os.makedirs(save_dir, exist_ok=True)
            
            # 🌟 增量保存：分别保存 LoRA (safetensors) 和 自定义分类头
            model.backbone.dino.save_pretrained(save_dir)
            torch.save(model.detector.state_dict(), os.path.join(save_dir, "custom_head.pth"))
            
            print(f"🌟 发现最佳模型！增量参数已轻量化保存至: {save_dir}")

if __name__ == '__main__':
    main()