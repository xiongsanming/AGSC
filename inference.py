import os
import glob
import argparse
import torch
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm

# 引入昇腾 NPU 适配
import torch_npu
from torch_npu.contrib import transfer_to_npu

# 假设你的模型定义在之前的 train_baseline_npu.py 中
# 如果你的文件名不同，请相应修改 import
from train_baseline_npu import Baseline_Detector

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 1. 数据预处理与加载
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

class InferenceDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        self.image_paths = [
            p for p in glob.glob(os.path.join(image_dir, "*.*")) 
            if os.path.splitext(p)[-1].lower() in valid_exts
        ]
        self.transform = DynamicScaleCenterCrop(224)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        filename = os.path.basename(path)
        try:
            img = Image.open(path).convert('RGB')
            tensor = self.transform(img)
            return tensor, filename
        except Exception as e:
            print(f"⚠️ 无法读取图片 {filename}: {e}")
            # 如果某张图片损坏，返回全零张量，防止程序崩溃
            return torch.zeros((3, 224, 224)), filename

# ==========================================
# 2. 推理主循环
# ==========================================
def run_inference():
    parser = argparse.ArgumentParser()
    # 默认路径已修改为你截图中的路径
    parser.add_argument('--image_dir', type=str, default='/data/AGSC/example-s6/image')
    parser.add_argument('--backbone_path', type=str, required=True, help="DINOv3 预训练权重路径")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="baseline_checkpoint.pth 的路径")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device_id', type=int, default=4, help='使用的 NPU 卡号')
    args = parser.parse_args()

    # 设置 NPU 设备
    if torch.npu.is_available():
        torch.npu.set_device(args.device_id)
        device = torch.device(f'npu:{args.device_id}')
        print(f"✅ 成功挂载昇腾 NPU 设备: {device}")
    else:
        device = torch.device('cpu')
        print("❌ 未检测到 NPU 设备，降级使用 CPU")

    print(f"📂 正在扫描文件夹: {args.image_dir}")
    dataset = InferenceDataset(args.image_dir)
    if len(dataset) == 0:
        print("⚠️ 未找到任何有效图片，请检查路径！")
        return
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("🚀 初始化基线模型...")
    model = Baseline_Detector(dino_path=args.backbone_path, feature_dim=1280).to(device)

    print(f"📦 正在加载权重: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model']) 
    model.eval()

    print("\n" + "="*60)
    print(f"{'Filename':<30} | {'Prediction':<10} | {'Fake Confidence':<15}")
    print("="*60)

    results = []

    with torch.no_grad():
        for imgs, filenames in tqdm(loader, desc="推理中"):
            imgs = imgs.to(device)
            # 昇腾 NPU 混合精度
            with torch.amp.autocast('npu'):
                logits = model(imgs)
                # 使用 Softmax 将输出转换为 0~1 的概率分布
                probs = torch.softmax(logits, dim=1)
                
            # 提取类别 1 (Fake) 的概率作为置信度
            fake_probs = probs[:, 1].cpu().numpy()
            
            for i in range(len(filenames)):
                fake_prob = fake_probs[i]
                # 设定阈值 0.5，大于 0.5 判定为 Fake，否则为 Real
                pred_label = "Fake" if fake_prob > 0.5 else "Real"
                
                # 打印到控制台
                print(f"{filenames[i]:<30} | {pred_label:<10} | {fake_prob*100:>6.2f}%")
                results.append((filenames[i], pred_label, fake_prob))

    print("="*60)
    print(f"🎉 推理完成！共处理 {len(results)} 张图片。")

if __name__ == '__main__':
    run_inference()