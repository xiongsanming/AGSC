from PIL import Image, ImageDraw
import os

def draw_and_save_crop_box(original_image_path: str, box_coords: tuple, output_path: str):
    with Image.open(original_image_path) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle(box_coords, outline="yellow", width=3)
        img.save(output_path)
        print(f"image with the crop box is generated: {output_path}")
