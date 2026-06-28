import os
import json
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
from peft import PeftModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- 务必确保这里的导入与你的模型定义文件一致 ---
from train_npu import Enhanced_Detector, DynamicScaleCenterCrop 

class InferenceDataset(Dataset):
    def __init__(self, image_dir):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
        self.image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir) 
            if f.lower().endswith(valid_extensions)
        ]
        self.transform = DynamicScaleCenterCrop(224)
        
    def __len__(self): 
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), path
        except Exception as e:
            print(f"⚠️ 警告: 无法读取图片 {path}, 错误信息: {e}")
            # 遇到坏图默认返回全零张量，并将在后续推理中给出默认预测
            return torch.zeros((3, 224, 224)), path

def inference():
    parser = argparse.ArgumentParser(description="Generate JSONL Submission")
    parser.add_argument('--image_dir', type=str, required=True, help="待推理的图片文件夹路径")
    parser.add_argument('--backbone_path', type=str, required=True, help="DINOv3 预训练模型路径")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="LoRA和分类头权重文件夹路径")
    parser.add_argument('--output_file', type=str, default='submission.jsonl', help="输出的 JSONL 文件名")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    # 增加一个阈值参数，方便你针对 F1 Score 调参
    parser.add_argument('--threshold', type=float, default=0.5, help="判定为生成的概率阈值")
    args = parser.parse_args()

    device = torch.device('npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu')
    print(f"🚀 初始化推理模型，运行设备: {device}")
    
    # 1. 初始化模型
    model = Enhanced_Detector(dino_path=args.backbone_path, feature_dim=1280, finetune=False).to(device)
    
    # 2. 加载 LoRA 与 分类头
    model.backbone.dino = PeftModel.from_pretrained(model.backbone.dino, args.checkpoint_path)
    head_path = os.path.join(args.checkpoint_path, "custom_head.pth")
    model.detector.load_state_dict(torch.load(head_path, map_location='cpu'))
    model.eval()

    # 3. 准备数据
    dataset = InferenceDataset(args.image_dir)
    if len(dataset) == 0:
        print("未找到图片，退出。")
        return
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"📝 开始推理并生成提交文件: {args.output_file} ...")

    # 4. 推理并直接写入 JSONL
    # 使用 'w' 模式和 'utf-8' 编码写入，完全符合提交要求
    with open(args.output_file, 'w', encoding='utf-8') as f:
        with torch.no_grad():
            for imgs, paths in tqdm(loader, desc="推理中"):
                imgs = imgs.to(device, non_blocking=True)
                
                with torch.amp.autocast('npu'):
                    logits = model(imgs)
                    # 假设 Class 1 是生成图片（正类）
                    probs = torch.softmax(logits, dim=1)[:, 1]
                
                probs = probs.cpu().numpy()
                
                for path, prob in zip(paths, probs):
                    filename = os.path.basename(path)
                    
                    # 🌟 核心逻辑：根据阈值转换为 "1" 或 "0" (必须是字符串)
                    is_generated = "1" if prob > args.threshold else "0"
                    
                    # 构建字典并转为 JSON 字符串写入
                    record = {
                        "image_name": filename,
                        "is_generated": is_generated
                    }
                    f.write(json.dumps(record) + "\n")

    print(f"✅ 推理完成！共处理 {len(dataset)} 张图片。结果已保存至 {args.output_file}")

if __name__ == '__main__':
    inference()
