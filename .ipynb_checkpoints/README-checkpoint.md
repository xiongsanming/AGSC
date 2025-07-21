# RINE 模型检测 Demo (含 AGSC 智能裁剪)

本项目是一个使用带有 LoRA 的 RINE 模型来检测图像是真实拍摄还是由 AI 生成的演示程序。

其主要特点是集成了一个**注意力引导的智能裁剪（Attention Guided Smart Cropping, AGSC）**模块。当处理高分辨率图像时，此模块能自动利用 CLIP 模型的注意力图定位图像中最显著的区域进行裁剪，然后再送入 RINE 模型进行判断，从而有效提升对大尺寸图像的检测效果。

## 项目结构

```
RINE-AGSC-Demo/
├── demo.py                 # 主执行脚本
├── agsc.py                 # AttentionCrop (AGSC) 实现
├── rine_model.py           # RINE 模型结构定义
├── utils.py                # 辅助函数
├── config.py               # 默认配置常量
├── demo_agsc_visual.ipynb  # 可视化 AGSC 效果的 Notebook
├── model_weights/          # 存放模型权重
├── images/                 # 存放待测试图像
├── requirements.txt        # 项目依赖
└── README.md               # 本说明文件
```

## 环境准备

1.  **克隆仓库**
    ```bash
    git clone <your-repo-url>
    cd RINE-AGSC-Demo
    ```

2.  **创建虚拟环境 (推荐)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

4.  **准备模型权重**
    请将您的 LoRA 适配器和 RINE Head 权重文件放入 `model_weights/` 目录，结构如下：
    ```
    model_weights/
    ├── lora_adapter_dir/      # 包含 adapter_config.json, adapter_model.bin 等
    └── rine_head_weights.pth  # Head 权重文件
    ```

## 如何使用

### 命令行执行

通过 `demo.py` 脚本对单个图像进行检测。

```bash
python demo.py \
  --image_path images/generated_101.png \
  --lora_adapter_dir model_weights/lora_adapter_dir/ \
  --head_weights_path model_weights/rine_head_weights.pth
```

**智能裁剪 (AGSC) 行为:**
-   **自动启用**: 如果输入图像的宽度或高度 `>= 512` 像素，脚本会自动启用 `attention` 裁剪模式。
-   **手动指定**: 您可以通过 `--crop_type` 参数来强制指定裁剪模式，例如：
    -   使用 AGSC: `--crop_type attention`
    -   禁用裁剪: `--crop_type none`
    -   使用中心裁剪: `--crop_type center`

### 输出

脚本执行后，将会在控制台打印出检测结果。如果使用了 `attention` 裁剪，还会额外生成一张在原图上标注了裁剪框的图片，文件名类似于 `original_name_crop_vis.png`。

```
--- 检测结果 ---
文件: generated_101.png
判断: AI生成图像 (Generated)
置信度: 0.9876
(模型原始输出 [真实图像概率]: 0.0124)
已生成带裁剪框的图像: images/generated_101_crop_vis.png

Demo 执行完毕。
```

### 可视化 AGSC 效果

如果您想直观地理解 AGSC 是如何工作的，或者想调试裁剪效果，请使用 Jupyter Notebook。

1.  启动 Jupyter:
    ```bash
    jupyter notebook
    ```
2.  在打开的浏览器页面中，点击并运行 `demo_agsc_visual.ipynb`。

这个 Notebook 将会加载一张图片，并清晰地展示出 AGSC 计算出的裁剪框位置以及最终裁剪出的图像。