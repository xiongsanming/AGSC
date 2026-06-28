import os
import glob
import pyarrow as pa
import io
import argparse
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF

from transformers import AutoModel
from peft import LoraConfig, get_peft_model

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 🌟 设备自适应初始化 (支持 NPU / GPU / CPU)
# ==========================================
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

def get_device():
    if HAS_NPU and torch.npu.is_available():
        return torch.device('npu'), 'npu'
    elif torch.cuda.is_available():
        return torch.device('cuda'), 'cuda'
    return torch.device('cpu'), 'cpu'

DEVICE, AMP_DEVICE_TYPE = get_device()

# ==========================================
# 1. 图像预处理与模型结构定义
# ==========================================
class DynamicScaleCenterCrop:
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        w, h = img.size
        scale = self.size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = TF.resize(img, (new_h, new_w), antialias=True)
        return TF.center_crop(img, self.size)

class Norm(nn.Module):
    def forward(self, x):
        return TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class AdvancedPatchClassifier(nn.Module):
    def __init__(self, feature_dim=1280, num_classes=2):
        super().__init__()
        self.attn_pool = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, patch_features):
        B = patch_features.shape[0]
        query = self.pool_query.expand(B, -1, -1)
        attn_out, _ = self.attn_pool(query, patch_features, patch_features)
        pooled_feat = attn_out.squeeze(1)
        return self.fc(pooled_feat)

class Enhanced_Detector(nn.Module):
    def __init__(self, dino_path, feature_dim=1280, finetune=True):
        super().__init__()
        self.norm = Norm()
        
        class DINOWrapper(nn.Module):
            def __init__(self, path):
                super().__init__()
                self.dino = AutoModel.from_pretrained(path)
            def forward(self, x):
                outputs = self.dino(x)
                return outputs.last_hidden_state[:, 1:]
                
        self.backbone = DINOWrapper(dino_path)
        
        if finetune:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.1,
                bias="none"
            )
            self.backbone.dino = get_peft_model(self.backbone.dino, lora_config)
            
        self.detector = AdvancedPatchClassifier(feature_dim=feature_dim)
        
    def forward(self, x):
        x = self.norm(x)
        features = self.backbone(x)
        return self.detector(features)

# ==========================================
# 2. CAID Arrow 数据集解析
# ==========================================
class CAIDArrowDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.tables = []
        self.offsets = []
        self.total_length = 0
        
        arrow_paths = glob.glob(os.path.join(root_dir, '**', 'test.arrow'), recursive=True)
        if not arrow_paths:
            raise ValueError(f"在 {root_dir} 中未找到 test.arrow 文件！")

        print(f"📦 加载 {len(arrow_paths)} 个 Arrow 文件 (内存映射)...")
        for path in arrow_paths:
            mmap = pa.memory_map(path, 'r')
            table = pa.ipc.open_file(mmap).read_all()
            self.tables.append(table)
            self.offsets.append(self.total_length)
            self.total_length += table.num_rows

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

        img_bytes = table['image'][local_idx].as_buffer()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # 🌟 核心：统一标签，0=真实照片，1=AI生成
        original_label = table['label'][local_idx].as_py()
        label = 1 - original_label 

        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)

# ==========================================
# 3. 训练主函数
# ==========================================
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help='CAIDBenchmark 根目录')
    parser.add_argument('--backbone_path', type=str, required=True, help='DINOv3 预训练路径')
    parser.add_argument('--output_dir', type=str, default='./weight/best_lora_model')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    print(f"🚀 初始化训练，运行设备: {DEVICE}")

    # 数据准备
    transform = DynamicScaleCenterCrop(224)
    full_dataset = CAIDArrowDataset(root_dir=args.dataset_root, transform=transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # 模型与优化器
    model = Enhanced_Detector(dino_path=args.backbone_path, finetune=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    best_auc = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            with torch.amp.autocast(AMP_DEVICE_TYPE):
                logits = model(imgs)
                loss = criterion(logits, labels)
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 验证阶段
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Eval]"):
                imgs = imgs.to(DEVICE)
                with torch.amp.autocast(AMP_DEVICE_TYPE):
                    logits = model(imgs)
                    probs = torch.softmax(logits, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(labels.numpy())
                
        val_probs = np.array(val_probs)
        val_labels = np.array(val_labels)
        
        val_acc = accuracy_score(val_labels, (val_probs > 0.5).astype(int))
        try: val_auc = roc_auc_score(val_labels, val_probs)
        except ValueError: val_auc = 0.0
        
        print(f"📊 Epoch {epoch+1} - Loss: {train_loss/len(train_loader):.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")

        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            print(f"⭐ 发现新 Best AUC: {best_auc:.4f}，正在保存模型...")
            model.backbone.dino.save_pretrained(args.output_dir)
            torch.save(model.detector.state_dict(), os.path.join(args.output_dir, "custom_head.pth"))

if __name__ == '__main__':
    train()