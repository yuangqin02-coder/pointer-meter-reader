# Reproduction Guide / 复刻指南

This guide explains how to reproduce the full pipeline on your own computer.
本指南说明如何在你自己的电脑上完整复现本项目流程。

---

## 0. Important Note / 重要提示

The source code files still contain the original absolute paths from the author's Linux development environment (e.g., `/home/qya/...`).
**You must replace these paths with your own paths before running.**

源码文件中仍保留了原作者 Linux 开发环境的绝对路径（例如 `/home/qya/...`）。
**运行前必须将这些路径替换为你自己电脑上的路径。**

Files containing hard-coded paths / 包含硬编码路径的文件：

- `用到的代码/predict_1111.py`
- `用到的代码/separate_1111.py`
- `用到的代码/990.py`
- `用到的代码/read_copy_bai_new.py`

---

## 1. Environment / 环境

### 1.1 Clone the repository / 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/pointer-meter-reader.git
cd pointer-meter-reader
```

### 1.2 Create a virtual environment / 创建虚拟环境

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 1.3 Install dependencies / 安装依赖

```bash
pip install -r requirements.txt
```

Main dependencies / 主要依赖：

- `numpy`
- `opencv-python`
- `ultralytics` (YOLOv8)

### 1.4 YOLO model weight / YOLO 权重文件

✅ The pre-trained YOLO model is **already included** in `yolo最新权重文件/best.pt`.
✅ 预训练的 YOLO 模型**已包含**在 `yolo最新权重文件/best.pt` 中。

---

## 2. Data Preparation / 数据准备

Place your raw images in a folder, for example / 将原始图片放入一个文件夹，例如：

```text
your_data/
└── raw/
    ├── 1-1.jpg
    ├── 1-2.jpg
    └── ...
```

You also need / 你还需要：

- ✅ **YOLO model weight** / YOLO 权重文件：**Already in `yolo最新权重文件/best.pt`**
- A template image for calibration and alignment (for example, a front-facing meter image).
  一张用于标定和图像摆正的模板图（例如一张正面仪表图）。

---

## 3. Calibration / 标定

Calibration creates `calibration_data.txt`, which tells the reader how pointer angles map to real values.
标定生成 `calibration_data.txt`，用于告诉识别程序指针角度如何对应真实读数。

1. Open `用到的代码/990.py`.
2. Change `IMG_PATH` to your template image:

```python
IMG_PATH = r"path/to/your/template.jpg"
```

3. Run / 运行：

```bash
cd 用到的代码
python 990.py
```

4. In the window that appears / 在弹出的窗口中：
   - Press `c` to start calibration / 按 `c` 开始标定。
   - Click the **center** of the gauge / 点击仪表圆心。
   - Click **32 points** along the pointer path from minimum to maximum value / 从最小值到最大值依次点击 32 个指针位置。
   - The program saves `calibration_data.txt` automatically / 程序会自动保存 `calibration_data.txt`。

The calibration file format is / 标定文件格式为：

```text
CENTER,x,y
0.0000,angle_rad_0
0.0516,angle_rad_1
...
1.6000,angle_rad_31
```

Make sure `calibration_data.txt` exists in the working directory when you run the reader.
运行识别程序时，请确保 `calibration_data.txt` 位于工作目录下。

---

## 4. YOLO Detection / YOLO 检测

1. Open `用到的代码/predict_1111.py`.
2. The YOLO model path already points to `../yolo最新权重文件/best.pt`.
   Replace only the `source` and `project` paths with your own:

```python
model = YOLO("../yolo最新权重文件/best.pt")  # ✅ Already configured

results = model.predict(
    source="path/to/your/raw/images",          # ⚠️ Change this to your input images folder
    imgsz=640,
    conf=0.55,
    save=True,
    save_txt=True,
    project="path/to/your/output"              # ⚠️ Change this to your output folder
)
```

3. Run / 运行：

```bash
cd 用到的代码
python predict_1111.py
```

Outputs / 输出：

- Detection images / 检测图：`path/to/your/output/predict/`
- Label files / 标签文件：`path/to/your/output/predict/labels/*.txt`

Each `.txt` file contains one line per detected object in YOLO format / 每个 `.txt` 文件包含一行或多行 YOLO 格式结果：

```text
class_id x_center y_center width height
```

Values are normalized to `[0, 1]` / 数值已归一化到 `[0, 1]`。

---

## 5. Crop Meter Regions / 裁剪单表区域

1. Open `用到的代码/separate_1111.py`.
2. Replace the paths with your own:

```python
img_path   = 'path/to/your/output/predict'
label_path = 'path/to/your/output/predict/labels'
save_path  = 'path/to/your/cropped'
```

3. Run / 运行：

```bash
cd 用到的代码
python separate_1111.py
```

Cropped images will be saved as / 裁剪后的图片命名为：

```text
original_name_1.jpg
original_name_2.jpg
...
```

---

## 6. Read Values / 读数识别

### 6.1 Single image / 单张图片

1. Open `用到的代码/read_copy_bai_new.py`.
2. Change these lines / 修改以下行：

```python
IMG_PATH      = r"path/to/your/template.jpg"       # 模板图
TEST_IMG_PATH = r"path/to/your/test_image.jpg"     # 待识别图
```

3. Make sure `calibration_data.txt` is in the `用到的代码/` directory.
   确保 `calibration_data.txt` 位于 `用到的代码/` 目录下。

4. Run / 运行：

```bash
cd 用到的代码
python read_copy_bai_new.py
```

Outputs / 输出到 `debug_output/`：

- `01_original.jpg`
- `02_aligned_to_template.jpg`
- `03_result_on_aligned.jpg`
- `04_pointer_mask.jpg`
- `06_low_light_enhanced.jpg` (only for low-light images)

### 6.2 Batch processing / 批量处理

```bash
cd 用到的代码
python batch_read_copy_bai_copy.py path/to/your/cropped/images
```

Optional arguments / 可选参数：

```bash
python batch_read_copy_bai_copy.py path/to/cropped \
    --template path/to/template.jpg \
    --output-root path/to/results \
    --include-template
```

Results / 结果：

- One folder per image / 每张图片一个子文件夹，包含：
  - `01_original.jpg`
  - `02_aligned_to_template.jpg`
  - `03_result_on_aligned.jpg`
  - `04_pointer_mask.jpg`
  - `05_low_light_enhanced.jpg` (if applicable)
- `summary.csv` with columns / 列包括：`image`, `value`, `angle_rad`, `success`, `output_dir`, `message`

---

## 7. Compare Multiple Methods / 多方法对比

Run / 运行：

```bash
cd 用到的代码
python batch_compare_readers.py path/to/your/cropped/images \
    --ground-truth path/to/your/summary.csv
```

If `summary.csv` is in the image directory, it will be detected automatically.
如果 `summary.csv` 在图片文件夹中，程序会自动识别。

Outputs / 输出到 `reader_comparison_results_2/`：

- `summary.csv` — per image, per method / 每张图片、每种方法的结果
- `metrics.csv` — aggregated metrics / 聚合指标
- Subfolders for each method and each image / 每种方法、每张图片的子文件夹

---

## 8. Typical Directory Layout for Reproduction / 推荐的复刻目录结构

```text
your_project/
├── best.pt                           # YOLO model (you provide)
├── calibration_data.txt              # generated by 990.py
├── raw_images/                       # your raw photos
├── detected/                         # YOLO outputs (predict_1111.py)
│   ├── predict/
│   │   ├── image1.jpg
│   │   └── labels/
│   │       └── image1.txt
├── cropped/                          # single meter images (separate_1111.py)
├── aligned_results/                  # batch_read_copy_bai_copy.py output
└── comparison_results/               # batch_compare_readers.py output
```

---

## 9. Troubleshooting / 常见问题

### Q1: `cv2` cannot be imported / 无法导入 `cv2`

Make sure you installed `opencv-python` in the virtual environment.
确保在虚拟环境中安装了 `opencv-python`。

```bash
pip install opencv-python
```

### Q2: `calibration_data.txt not found` / 找不到标定文件

Run `990.py` first, or copy an existing calibration file into the working directory.
先运行 `990.py` 生成标定文件，或将已有的标定文件复制到工作目录。

### Q3: YOLO detection produces no labels / YOLO 没有输出标签

Check that / 检查：

- `best.pt` exists and is a valid YOLOv8 model / `best.pt` 存在且是有效的 YOLOv8 模型。
- `conf=0.55` is not too high for your images / `conf=0.55` 对你的图片不会过高。
- The input path is correct / 输入路径正确。

### Q4: Alignment fails for some images / 部分图片摆正失败

The reader automatically falls back to Hough circle detection or simple resizing.
If most images fail, check that the template and test images are similar enough in appearance.
识别程序会自动回退到 Hough 圆检测或简单缩放。如果大部分图片失败，请检查模板图与测试图是否足够相似。

### Q5: Results differ from the paper / 结果与论文不一致

The current code is an engineering implementation that uses ORB + HSV segmentation + multi-anchor interpolation.
The reference paper additionally describes YOLOv8-pose keypoint detection for pointer extraction.
Your results will depend on your model, calibration, and image quality.

当前代码是工程实现版本，使用 ORB + HSV 分割 + 多锚点插值。
参考论文还描述了基于 YOLOv8-pose 关键点的指针提取方法。
实际结果取决于你的模型、标定质量和图片质量。

---

## 10. Minimal Example / 最小示例

```bash
# 1. Install
cd pointer-meter-reader
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 2. Calibrate
cd 用到的代码
# Edit 990.py: set IMG_PATH to your template.jpg
python 990.py
# Press 'c', click center, click 32 points.

# 3. Detect
cd 用到的代码
# Edit predict_1111.py: set model path, source, and project
python predict_1111.py

# 4. Crop
cd 用到的代码
# Edit separate_1111.py: set img_path, label_path, save_path
python separate_1111.py

# 5. Read
cd 用到的代码
python batch_read_copy_bai_copy.py path/to/cropped
```

---

If you still cannot reproduce the results, please open an issue with / 如果仍无法复现，请提交 issue 并附带：

- Your operating system and Python version / 操作系统和 Python 版本
- The exact command you ran / 运行的具体命令
- The full error message / 完整错误信息
- A few example images (if possible) / 几张示例图片（如果允许）
