# AGSC of AI-Generated Image Detection

Use `demo.py` script to detect a single image, with the provided weights in folder model_weights.

```bash
python demo.py  --lora_adapter_dir model_weights/ --head_weights_path model_weights/head_weights.pth --image_path test_img.png 
```

###  visualization AGSC

run `agsc_visual.ipynb` will load an image and display the cropping box position calculated by AGSC, as well as the final cropped image.
