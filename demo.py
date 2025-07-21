import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
import sys

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from peft import PeftModel

import config
from CLIP_model import LoRACLIP
from agsc import AttentionCrop
from utils import draw_and_save_crop_box

def predict_single_image(model, image_path, device, clip_model_name, args_cli):
    model.eval()
    crop_box_coords = None
    image = Image.open(image_path).convert("RGB")
    clip_image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    print("AGSC is enabled")
    pre_crop_transform = AttentionCrop(
        output_size=args_cli.attention_crop_output_size, clip_model_name=clip_model_name, device=device,
        k_top_patches=args_cli.attn_crop_k_top_patches, num_early_layers_to_aggregate=args_cli.attn_crop_num_early_layers,
        num_layers_for_highlight_counting=args_cli.attn_crop_num_highlight_layers, num_top_patches_for_centroid=args_cli.attn_crop_num_top_centroid_patches
    )
    image, crop_box_coords = pre_crop_transform(image)
    processed_output = clip_image_processor(images=image, return_tensors="pt")
    pixel_values = processed_output['pixel_values']

    with torch.no_grad():
        inputs = pixel_values.to(device)
        logits, _ = model(pixel_values=inputs)
        logits = logits.squeeze(-1)
        probability_real = torch.sigmoid(logits.float()).item()
    
    return probability_real, crop_box_coords


def main():
    parser = argparse.ArgumentParser(description="Demo for detecting a single image")
    parser.add_argument('--image_path', type=str, required=True, help="image_path")
    parser.add_argument('--lora_adapter_dir', type=str, required=True, help="LoRA adapter path")
    parser.add_argument('--head_weights_path', type=str, required=True, help="Head path")
    parser.add_argument('--clip_model_name', type=str, default=config.CLIP_MODEL_NAME_DEFAULT)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--num_intermediate_layers', type=int, default=config.NUM_INTERMEDIATE_LAYERS_DEFAULT)
    parser.add_argument('--projection_dim', type=int, default=config.PROJECTION_DIM_DEFAULT)
    parser.add_argument('--num_projection_layers', type=int, default=config.NUM_PROJECTION_LAYERS_DEFAULT)
    parser.add_argument('--dropout_rate', type=float, default=config.DROPOUT_RATE_DEFAULT)
    parser.add_argument('--crop_type', type=str, default='attention')
    parser.add_argument('--attention_crop_output_size', type=int, default=config.ATTENTION_CROP_OUTPUT_SIZE_DEFAULT)
    parser.add_argument('--attn_crop_k_top_patches', type=int, default=config.ATTN_CROP_K_TOP_PATCHES_DEFAULT)
    parser.add_argument('--attn_crop_num_early_layers', type=int, default=config.ATTN_CROP_NUM_EARLY_LAYERS_TO_AGGREGATE_DEFAULT)
    parser.add_argument('--attn_crop_num_highlight_layers', type=int, default=config.ATTN_CROP_NUM_HIGHLIGHT_LAYERS_DEFAULT)
    parser.add_argument('--attn_crop_num_top_centroid_patches', type=int, default=config.ATTN_CROP_NUM_TOP_CENTROID_PATCHES_DEFAULT)

    args = parser.parse_args()
    DEVICE = args.device
    print(f"tested image: {args.image_path}")
    print(f"LoRA adapter directory: {args.lora_adapter_dir}")
    print(f"Head weights: {args.head_weights_path}")
    print(f"\nloading CLIP model ({args.clip_model_name}) and lora adapter ({args.lora_adapter_dir})...")
    base_vision_model_full = CLIPVisionModelWithProjection.from_pretrained(args.clip_model_name)
    base_vision_model = base_vision_model_full.vision_model.to(DEVICE)
    lora_vision_model_loaded = PeftModel.from_pretrained(base_vision_model, args.lora_adapter_dir).to(DEVICE)

    model = LoRACLIP(
        lora_vision_model=lora_vision_model_loaded, n_intermediate_layers_to_use=args.num_intermediate_layers,
        proj_dim=args.projection_dim, n_proj_layers=args.num_projection_layers, dropout_rate=args.dropout_rate
    ).to(DEVICE)

    head_state = torch.load(args.head_weights_path, map_location=DEVICE)
    model.q1.load_state_dict(head_state['q1'])
    model.tie_weights.data.copy_(head_state['tie_weights'])
    model.q2.load_state_dict(head_state['q2'])
    model.classification_head.load_state_dict(head_state['classification_head'])
    
    print("\n--- detecting ---")
    prob_real, crop_coords = predict_single_image(model, args.image_path, DEVICE, args.clip_model_name, args)

    if prob_real is not None:
        print("\n--- detecting results ---")
        judgement = "Real" if prob_real > 0.5 else "AI-Generated"
        confidence = prob_real if prob_real > 0.5 else 1 - prob_real
        
        print(f"file: {os.path.basename(args.image_path)}")
        print(f"judgement: {judgement}")
        print(f"real image probability: {prob_real:.4f})")

        if crop_coords:
            base, ext = os.path.splitext(args.image_path)
            output_filename = f"{base}_crop_vis{ext}"
            draw_and_save_crop_box(args.image_path, crop_coords, output_filename)

if __name__ == "__main__":
    main()