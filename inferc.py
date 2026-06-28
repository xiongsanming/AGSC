import os
import json
import argparse
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel

# 从 train.py 导入核心组件（确保 train.py 和此脚本在同一目录下）
from train import Enhanced_Detector, DynamicScaleCenterCrop, get_device

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEVICE, AMP_DEVICE_TYPE = get_device()

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
            tqdm.write(f"⚠️ 警告: 无法读取图片 {path}, 已跳过处理。报错: {e}")
            return torch.zeros((3, 224, 224)), path

def inference():
    parser = argparse.ArgumentParser(description="生成比赛提交文件 (JSONL)")
    parser.add_argument('--image_dir', type=str, required=True, help="测试集图片文件夹路径")
    parser.add_argument('--backbone_path', type=str, required=True, help="DINOv3 预训练模型路径")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="LoRA和分类头权重文件夹路径")
    parser.add_argument('--output_file', type=str, default='submission.jsonl', help="输出的文件名")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--threshold', type=float, default=0.5, help="判定为生成的概率阈值 (影响F1)")
    args = parser.parse_args()

    print(f"🚀 初始化推理模型，运行设备: {DEVICE}")
    
    # 初始化模型结构 (finetune=False)
    model = Enhanced_Detector(dino_path=args.backbone_path, finetune=False).to(DEVICE)
    
    # 拼装 LoRA 与 自定义分类头
    model.backbone.dino = PeftModel.from_pretrained(model.backbone.dino, args.checkpoint_path)
    head_path = os.path.join(args.checkpoint_path, "custom_head.pth")
    # map_location='cpu' 确保 GPU/NPU 权重可以互相兼容加载
    model.detector.load_state_dict(torch.load(head_path, map_location='cpu'))
    model.eval()

    dataset = InferenceDataset(args.image_dir)
    if len(dataset) == 0:
        print("❌ 未找到图片，程序退出。")
        return
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"📝 开始推理并生成提交文件: {args.output_file} ...")

    # 推理并实时写入 JSONL
    with open(args.output_file, 'w', encoding='utf-8') as f:
        with torch.no_grad():
            for imgs, paths in tqdm(loader, desc="推理中"):
                imgs = imgs.to(DEVICE, non_blocking=True)
                
                with torch.amp.autocast(AMP_DEVICE_TYPE):
                    logits = model(imgs)
                    probs = torch.softmax(logits, dim=1)[:, 1]
                
                probs = probs.cpu().numpy()
                
                for path, prob in zip(paths, probs):
                    filename = os.path.basename(path)
                    
                    # 按照阈值二值化结果：1=生成，0=真实照片
                    is_generated = "1" if prob > args.threshold else "0"
                    
                    record = {
                        "image_name": filename,
                        "is_generated": is_generated
                    }
                    f.write(json.dumps(record) + "\n")

    print(f"✅ 推理完成！共处理 {len(dataset)} 张图片。结果已保存至 {args.output_file}")

if __name__ == '__main__':
    inference()