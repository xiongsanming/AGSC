import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from collections import Counter

from transformers import CLIPVisionModelWithProjection, CLIPProcessor

from config import (
    ATTN_CROP_K_TOP_PATCHES_DEFAULT,
    ATTN_CROP_NUM_EARLY_LAYERS_TO_AGGREGATE_DEFAULT,
    ATTN_CROP_NUM_HIGHLIGHT_LAYERS_DEFAULT,
    ATTN_CROP_NUM_TOP_CENTROID_PATCHES_DEFAULT
)

class AttentionCrop(nn.Module):
    def __init__(self, output_size, clip_model_name, device,
                 k_top_patches=ATTN_CROP_K_TOP_PATCHES_DEFAULT,
                 num_early_layers_to_aggregate=ATTN_CROP_NUM_EARLY_LAYERS_TO_AGGREGATE_DEFAULT,
                 num_layers_for_highlight_counting=ATTN_CROP_NUM_HIGHLIGHT_LAYERS_DEFAULT,
                 num_top_patches_for_centroid=ATTN_CROP_NUM_TOP_CENTROID_PATCHES_DEFAULT
                 ):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self.device = device
        try:
            # 使用 eager attention implementation
            self.attn_model = CLIPVisionModelWithProjection.from_pretrained(
                clip_model_name, attn_implementation="eager"
            ).to(self.device)
            self.attn_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.attn_model.eval()
        except Exception as e:
            print(f"AttentionCrop: Error loading model {clip_model_name}: {e}")
            raise
        self.patch_size = self.attn_model.config.patch_size
        self.total_model_layers = self.attn_model.config.num_hidden_layers
        self.k_top_patches = k_top_patches
        self.num_early_layers_to_aggregate = min(num_early_layers_to_aggregate, self.total_model_layers)
        self.num_layers_for_highlight_counting = min(num_layers_for_highlight_counting, self.total_model_layers)
        self.num_top_patches_for_centroid = num_top_patches_for_centroid

    def forward(self, img: Image.Image) -> (Image.Image, tuple or None):
        raw_image_pil = img.copy()
        try:
            # === 此处为原脚本中的完整 AttentionCrop 逻辑，保持不变 ===
            inputs = self.attn_processor(images=raw_image_pil, return_tensors="pt", padding=True)
            pixel_values = inputs["pixel_values"].to(self.device)
            if pixel_values.shape[2] < self.patch_size or pixel_values.shape[3] < self.patch_size:
                return transforms.CenterCrop(self.output_size)(raw_image_pil), None
            with torch.no_grad():
                outputs = self.attn_model(pixel_values, output_attentions=True)
                all_layer_attentions = outputs.attentions
            if not all_layer_attentions:
                return transforms.CenterCrop(self.output_size)(raw_image_pil), None
            processed_height, processed_width = pixel_values.shape[2], pixel_values.shape[3]
            num_patches_h, num_patches_w = processed_height // self.patch_size, processed_width // self.patch_size
            num_patches_total = num_patches_h * num_patches_w
            if num_patches_total == 0: return transforms.CenterCrop(self.output_size)(raw_image_pil), None
            aggregated_cls_attention_scores = torch.zeros(num_patches_total, device='cpu')
            valid_layers_for_aggregation = 0
            for layer_idx in range(self.num_early_layers_to_aggregate):
                if layer_idx >= len(all_layer_attentions): continue
                current_layer_attentions_cpu = all_layer_attentions[layer_idx].cpu()
                cls_attention_this_layer = current_layer_attentions_cpu[0, :, 0, 1:]
                if cls_attention_this_layer.shape[1] != num_patches_total: continue
                avg_cls_attention_this_layer_for_agg = cls_attention_this_layer.mean(dim=0)
                aggregated_cls_attention_scores += avg_cls_attention_this_layer_for_agg
                valid_layers_for_aggregation +=1
            if valid_layers_for_aggregation > 0: aggregated_cls_attention_scores /= valid_layers_for_aggregation
            else: return transforms.CenterCrop(self.output_size)(raw_image_pil), None
            patch_highlight_counts = Counter()
            for layer_idx in range(self.num_layers_for_highlight_counting):
                if layer_idx >= len(all_layer_attentions): continue
                current_layer_attentions_cpu = all_layer_attentions[layer_idx].cpu()
                cls_attention_this_layer = current_layer_attentions_cpu[0, :, 0, 1:]
                if cls_attention_this_layer.shape[1] != num_patches_total: continue
                avg_cls_attention_this_layer = cls_attention_this_layer.mean(dim=0)
                current_k_for_highlight = min(self.k_top_patches, avg_cls_attention_this_layer.shape[0])
                if current_k_for_highlight == 0: continue
                _, top_k_indices_this_layer = torch.topk(avg_cls_attention_this_layer, current_k_for_highlight)
                for p_idx in top_k_indices_this_layer.numpy(): patch_highlight_counts[p_idx] += 1
            all_once_highlighted_patch_indices = list(patch_highlight_counts.keys())
            if not all_once_highlighted_patch_indices: return transforms.CenterCrop(self.output_size)(raw_image_pil), None
            sorted_patches = sorted(all_once_highlighted_patch_indices, key=lambda p_idx: (patch_highlight_counts[p_idx], aggregated_cls_attention_scores[p_idx].item()), reverse=True)
            patches_for_centroid = sorted_patches[:self.num_top_patches_for_centroid]
            if not patches_for_centroid: return transforms.CenterCrop(self.output_size)(raw_image_pil), None
            original_scores_for_centroid_calc = torch.tensor([aggregated_cls_attention_scores[p_idx].item() for p_idx in patches_for_centroid], dtype=torch.float32, device='cpu')
            final_weights_for_centroid = torch.empty(0, dtype=torch.float32, device='cpu')
            if original_scores_for_centroid_calc.numel() > 0:
                min_val, max_val = original_scores_for_centroid_calc.min(), original_scores_for_centroid_calc.max()
                if (max_val - min_val).item() < 1e-6: normalized_scores_input = torch.zeros_like(original_scores_for_centroid_calc)
                else: normalized_scores_input = (original_scores_for_centroid_calc - min_val) / (max_val - min_val)
                final_weights_for_centroid = torch.softmax(normalized_scores_input, dim=0)
            weighted_sum_x_px, weighted_sum_y_px, sum_of_weights = 0.0, 0.0, 0.0
            center_x_resized, center_y_resized = processed_width / 2.0, processed_height / 2.0
            if final_weights_for_centroid.numel() > 0 and final_weights_for_centroid.sum().item() > 1e-6 :
                for i, patch_idx_flat in enumerate(patches_for_centroid):
                    patch_row, patch_col = patch_idx_flat // num_patches_w, patch_idx_flat % num_patches_w
                    current_patch_center_x, current_patch_center_y = (patch_col + 0.5) * self.patch_size, (patch_row + 0.5) * self.patch_size
                    weight = final_weights_for_centroid[i].item()
                    weighted_sum_x_px += current_patch_center_x * weight
                    weighted_sum_y_px += current_patch_center_y * weight
                    sum_of_weights += weight
                if sum_of_weights > 1e-6: center_x_resized, center_y_resized = weighted_sum_x_px / sum_of_weights, weighted_sum_y_px / sum_of_weights
            scale_x, scale_y = raw_image_pil.width / processed_width, raw_image_pil.height / processed_height
            center_x_original, center_y_original = center_x_resized * scale_x, center_y_resized * scale_y
            crop_width, crop_height = self.output_size
            crop_x0, crop_y0 = center_x_original - crop_width / 2.0, center_y_original - crop_height / 2.0
            crop_x0, crop_y0 = max(0.0, crop_x0), max(0.0, crop_y0)
            if crop_x0 + crop_width > raw_image_pil.width: crop_x0 = float(raw_image_pil.width - crop_width)
            if crop_y0 + crop_height > raw_image_pil.height: crop_y0 = float(raw_image_pil.height - crop_height)
            crop_x0, crop_y0 = max(0.0, crop_x0), max(0.0, crop_y0)
            final_x0, final_y0 = int(round(crop_x0)), int(round(crop_y0))
            final_x1, final_y1 = final_x0 + crop_width, final_y0 + crop_height
            final_x1, final_y1 = min(final_x1, raw_image_pil.width), min(final_y1, raw_image_pil.height)
            final_x0, final_y0 = max(0, final_x1 - crop_width), max(0, final_y1 - crop_height)

            cropped_image = raw_image_pil.crop((final_x0, final_y0, final_x1, final_y1))
            if cropped_image.size != self.output_size:
                cropped_image = cropped_image.resize(self.output_size, Image.Resampling.LANCZOS)
            
            return cropped_image, (final_x0, final_y0, final_x1, final_y1)

        except Exception as e:
            print(f"AttentionCrop: An exception occurred: {e}. Falling back to CenterCrop.")
            return transforms.CenterCrop(self.output_size)(raw_image_pil), None