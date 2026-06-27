import os
import glob
import random
import argparse
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# ==========================================
# 昇腾 NPU 核心适配组件导入
# ==========================================
import torch_npu
from torch_npu.contrib import transfer_to_npu # 自动将常见 cuda 操作拦截并桥接到 npu

# 仅引入骨干网络和归一化模块
from models.mirror import DINO, Norm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 1. 简化的“直出分类”模型架构
# ==========================================
class DirectPatchClassifier(nn.Module):
    def __init__(self, feat_dim=1280, hidden_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, feat_tokens, feat_cls):
        pooled_tokens = feat_tokens.mean(dim=1)
        combined = torch.cat([pooled_tokens, feat_cls], dim=-1)
        return self.head(combined)

class Baseline_Detector(nn.Module):
    def __init__(self, dino_path, feature_dim=1280):
        super().__init__()
        self.norm = Norm(mode='imagenet')
        self.backbone = DINO(dino_path, finetune=True) 
        self.detector = DirectPatchClassifier(feat_dim=feature_dim)

    def forward(self, x):
        x = self.norm(x)
        feat_tokens, feat_cls = self.backbone(x)
        logits = self.detector(feat_tokens, feat_cls)
        return logits

# ==========================================
# 2. 数据处理与加载
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
        img = TF.center_crop(img, [self.crop_size, self.crop_size])
        return TF.to_tensor(img)

class BaselineDataset(Dataset):
    def __init__(self, data_list, transform):
        self.data_list = data_list
        self.transform = transform
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        path, label = self.data_list[idx]
        try:
            return self.transform(Image.open(path).convert('RGB')), torch.tensor(label, dtype=torch.long)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.data_list))

def build_datasets(data_root, seed=42):
    random.seed(seed)
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    real_paths = [p for p in glob.glob(os.path.join(data_root, "0_real", "*.*")) if os.path.splitext(p)[-1].lower() in valid_exts]
    fake_paths = [p for p in glob.glob(os.path.join(data_root, "1_fake", "*.*")) if os.path.splitext(p)[-1].lower() in valid_exts]
    
    real_data = [(p, 0) for p in real_paths]
    fake_data = [(p, 1) for p in fake_paths]
    random.shuffle(real_data)
    random.shuffle(fake_data)
    
    def split(data):
        n = len(data)
        return data[:int(n*0.7)], data[int(n*0.7):int(n*0.8)], data[int(n*0.8):]
    
    rt, rv, rts = split(real_data)
    ft, fv, fts = split(fake_data)
    
    train_data = rt + ft
    val_data = rv + fv
    random.shuffle(train_data)
    
    transform = DynamicScaleCenterCrop(224)
    return BaselineDataset(train_data, transform), BaselineDataset(val_data, transform)

# ==========================================
# 3. 训练主循环
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/agsc/DDA/todda/')
    parser.add_argument('--backbone_path', type=str, default='/root/autodl-tmp/cache/facebook/dinov3-vith16plus-pretrain-lvd1689m')
    parser.add_argument('--save_dir', type=str, default='./weight')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    # 新增参数：显式指定使用的 NPU 卡号
    parser.add_argument('--device_id', type=int, default=4, help='使用的 NPU 设备 ID，如 4 或 5')
    args = parser.parse_args()

    # 设置 NPU 设备
    if torch.npu.is_available():
        torch.npu.set_device(args.device_id)
        device = torch.device(f'npu:{args.device_id}')
        print(f"✅ 成功挂载昇腾 NPU 设备: {device}")
    else:
        device = torch.device('cpu')
        print("❌ 未检测到 NPU 设备，降级使用 CPU")
        
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("📦 正在加载并划分数据集...")
    train_dataset, val_dataset = build_datasets(args.data_root)
    # 在某些旧版昇腾环境下，pin_memory=True 可能会导致冲突，这里默认关闭
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    print("🚀 初始化基线模型 (DINOv3 LoRA + Direct MLP Classifier)...")
    model = Baseline_Detector(dino_path=args.backbone_path, feature_dim=1280).to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # 使用安全的昇腾混合精度 (NPU 上的 autocast 目标设备设为 'npu')
            with torch.amp.autocast('npu'):
                logits = model(imgs)
                loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        scheduler.step()

        # 验证评估
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
                imgs = imgs.to(device)
                with torch.amp.autocast('npu'):
                    logits = model(imgs)
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(labels.numpy())
                
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"Epoch {epoch+1} 验证集准确率: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, "baseline_checkpoint.pth")
            torch.save({'model': model.state_dict()}, save_path)
            print(f"🌟 发现最佳模型并保存至: {save_path}")

if __name__ == '__main__':
    main()