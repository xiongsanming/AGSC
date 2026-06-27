import os
import glob
import argparse
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# ==========================================
# 🌟 NPU 与组装加载核心库
# ==========================================
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

from transformers import AutoModel
from peft import LoraConfig, get_peft_model, PeftModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- （此处复用前面的模型定义：Norm, DINO, AdvancedPatchClassifier, Enhanced_Detector, DynamicScaleCenterCrop）---
# 务必确保这里的导入与 train_npu.py 保持一致
from train_npu import Norm, DINO, AdvancedPatchClassifier, Enhanced_Detector, DynamicScaleCenterCrop 

class InferenceDataset(Dataset):
    def __init__(self, image_dir):
        # 支持常见图片格式
        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
        self.image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir) 
            if f.lower().endswith(valid_extensions)
        ]
        if len(self.image_paths) == 0:
            print(f"⚠️ 在 {image_dir} 中没有找到支持的图片文件！")
            
        self.transform = DynamicScaleCenterCrop(224)
        
    def __len__(self): 
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            tensor_img = self.transform(img)
            return tensor_img, path
        except Exception as e:
            # 推理时如果图片损坏，返回全零张量并打印警告，避免错位
            print(f"⚠️ 警告: 无法读取图片 {path}, 错误信息: {e}")
            return torch.zeros((3, 224, 224)), path

def inference():
    parser = argparse.ArgumentParser(description="Image Inference Script")
    # 匹配截图中的参数名
    parser.add_argument('--image_dir', type=str, required=True, help="待推理的图片文件夹路径")
    parser.add_argument('--backbone_path', type=str, required=True, help="DINOv3 预训练模型路径")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="LoRA和分类头权重文件夹路径")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    # 🌟 NPU 设备指定
    device = torch.device('npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu')
    print(f"🚀 初始化推理模型，运行设备: {device}")
    
    # 初始化模型结构 (finetune=False)
    model = Enhanced_Detector(dino_path=args.backbone_path, feature_dim=1280, finetune=False).to(device)

    print(f"📦 正在拼装增量权重: {args.checkpoint_path}")
    
    # 1. 加载 LoRA 增量 (读取 adapter_config.json 和 .safetensors)
    model.backbone.dino = PeftModel.from_pretrained(model.backbone.dino, args.checkpoint_path)
    
    # 2. 加载自定义分类头 (custom_head.pth)
    head_path = os.path.join(args.checkpoint_path, "custom_head.pth")
    if not os.path.exists(head_path):
        raise FileNotFoundError(f"找不到分类头权重文件: {head_path}")
    model.detector.load_state_dict(torch.load(head_path, map_location='cpu'))
    
    model.eval()

    # 准备数据
    dataset = InferenceDataset(args.image_dir)
    if len(dataset) == 0:
        return
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("\n" + "="*70)
    print(f"{'Image Name':<45} | {'Confidence (Class 1)':<20}")
    print("="*70)

    results = []

    with torch.no_grad():
        for imgs, paths in tqdm(loader, desc="推理中"):
            imgs = imgs.to(device, non_blocking=True)
            
            # 🌟 NPU 混合精度推理
            with torch.amp.autocast('npu'):
                logits = model(imgs)
                # 假设 Class 1 代表 Fake/目标类别
                probs = torch.softmax(logits, dim=1)[:, 1]
            
            probs = probs.cpu().numpy()
            
            # 记录并打印结果
            for path, prob in zip(paths, probs):
                filename = os.path.basename(path)
                print(f"{filename:<45} | {prob:.4f} ({prob*100:>5.2f}%)")
                results.append((filename, prob))

    print("="*70)
    print(f"✅ 推理完成，共处理 {len(results)} 张图片。")
    
    # 如果需要，可以将结果保存到 csv 或 txt
    # with open('inference_results.csv', 'w') as f:
    #     f.write("filename,confidence\n")
    #     for name, prob in results:
    #         f.write(f"{name},{prob:.4f}\n")

if __name__ == '__main__':
    inference()