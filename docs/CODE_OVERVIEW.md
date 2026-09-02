# Code Overview / 代码说明

This document explains the purpose of each Python script in `用到的代码/`.

---

## File Map / 文件对应表

| Original filename | Purpose | Recommended new name |
|---|---|---|
| `990.py` | Interactive calibration tool | `calibrate.py` |
| `predict_1111.py` | YOLO inference for meter detection | `yolo_predict.py` |
| `separate_1111.py` | Crop single-meter images from YOLO outputs | `crop_from_yolo.py` |
| `read_copy_bai_new.py` | Core gauge reader class (`GaugeApp`) | `gauge_reader.py` |
| `batch_read_copy_bai_copy.py` | Batch reading for a folder of meter images | `batch_read.py` |
| `equal_angle_linear_reader.py` | Baseline reader: equal-angle linear mapping | `equal_angle_reader.py` |
| `traditional_hough_reader.py` | Baseline reader: Hough line detection | `hough_reader.py` |
| `batch_compare_readers.py` | Run and compare three readers | `compare_readers.py` |

---

## 1. `990.py` — Calibration Tool / 标定工具

What it does / 功能：

- Loads a template image.
- Lets the user click the gauge center and 32 pointer positions.
- Saves `calibration_data.txt`.

Key variables / 关键变量：

```python
IMG_PATH = r"1-1_1.jpg"          # template image
```

Usage / 用法：

```bash
python 990.py
# Press 'c' → click center → click 32 points
```

---

## 2. `predict_1111.py` — YOLO Detection / YOLO 检测

What it does / 功能：

- Loads a trained YOLOv8 model from `../yolo最新权重文件/best.pt`.
- Runs inference on raw images.
- Saves detection images and YOLO-format labels.

Paths to change / 需要修改的路径：

```python
model = YOLO("../yolo最新权重文件/best.pt")  # ✅ Already configured (no change needed)

results = model.predict(
    source="/home/qya/YOLO_Projects/Instrument/data/111",      # ⚠️ Change to your input folder
    ...
    project="/home/qya/YOLO_Projects/Instrument/data/111_out"  # ⚠️ Change to your output folder
)
```

Only `source` (input images) and `project` (output root) need to be changed by the user.
用户只需修改 `source`（输入图片文件夹）和 `project`（输出根目录）。

Usage / 用法：

```bash
python predict_1111.py
```

---

## 3. `separate_1111.py` — Crop from YOLO / 根据 YOLO 结果裁剪

What it does / 功能：

- Reads detection images and `labels/*.txt`.
- Crops each detected bounding box.
- Saves cropped single-meter images.

Hard-coded paths to change / 需要修改的硬编码路径：

```python
img_path   = '/home/qya/YOLO_Projects/Instrument/data/111_out/predict'
label_path = '/home/qya/YOLO_Projects/Instrument/data/111_out/predict/labels'
save_path  = '/home/qya/YOLO_Projects/Instrument/data/111_sep'
```

Usage / 用法：

```bash
python separate_1111.py
```

---

## 4. `read_copy_bai_new.py` — Core Reader / 核心识别类

What it does / 功能：

This is the main algorithm. It defines the `GaugeApp` class with these steps:

1. **Load calibration** from `calibration_data.txt`.
2. **Align** the test image to the template using ORB feature matching + RANSAC homography.
   - Falls back to Hough circle detection if ORB fails.
   - Falls back to simple resize if both fail.
3. **Preprocess** for low light if the mean brightness is below a threshold.
4. **Segment** the red pointer in HSV color space.
5. **Detect pointer tip** using connected components and radial analysis.
6. **Convert angle to value** using multi-anchor interpolation.
7. **Return** value, angle, mask, aligned image, tip, and optional low-light image.

Key variables to change / 需要修改的关键变量：

```python
IMG_PATH      = r"1-1_1.jpg"          # template image
TEST_IMG_PATH = r"3_3_1_3_1.jpg"      # single test image
DEBUG_OUTPUT_DIR = r"debug_output"    # output folder
```

Main method / 主要方法：

```python
value, angle, mask, corrected_img, tip, low_light_img = app.detect(test_img)
```

---

## 5. `batch_read_copy_bai_copy.py` — Batch Reading / 批量识别

What it does / 功能：

- Processes a whole folder of cropped meter images.
- Creates one output subfolder per image.
- Writes `summary.csv`.

Usage / 用法：

```bash
python batch_read_copy_bai_copy.py path/to/cropped/images
```

Options / 选项：

```bash
--template        # template image path
--output-root     # root output directory
--include-template
```

This script already uses command-line arguments; only the default template path is hard-coded.

---

## 6. `equal_angle_linear_reader.py` — Equal-Angle Baseline / 等角度线性基线

What it does / 功能：

- Inherits from `GaugeApp`.
- Replaces multi-anchor interpolation with simple equal-angle linear mapping between the first and last calibration angles.

Usage / 用法：

Imported by `batch_compare_readers.py`.

---

## 7. `traditional_hough_reader.py` — Hough Baseline / Hough 基线

What it does / 功能：

- Inherits from `GaugeApp`.
- Uses ORB alignment + HSV segmentation + HoughLinesP for pointer detection.
- Uses equal-angle linear mapping for value conversion.

Usage / 用法：

Imported by `batch_compare_readers.py`.

---

## 8. `batch_compare_readers.py` — Method Comparison / 方法对比

What it does / 功能：

- Runs three readers on the same folder:
  - `traditional_hough`
  - `equal_angle_linear`
  - `proposed_multi_anchor`
- Computes success rate, MAE, RMSE, max error, accuracy within 0.05, and average time.

Usage / 用法：

```bash
python batch_compare_readers.py path/to/cropped/images --ground-truth path/to/summary.csv
```

If `summary.csv` exists in the image directory, `--ground-truth` can be omitted.

---

## Dependency Graph / 依赖关系

```text
predict_1111.py                → Ultralytics YOLO
     ↓
separate_1111.py               → cv2, os
     ↓
read_copy_bai_new.py (GaugeApp) → cv2, numpy, math, os
     ↑
990.py                         → generates calibration_data.txt
     ↓
batch_read_copy_bai_copy.py    → GaugeApp
     ↓
equal_angle_linear_reader.py   → GaugeApp
traditional_hough_reader.py    → GaugeApp
     ↓
batch_compare_readers.py       → all three readers
```

---

## Notes for Refactoring / 重构建议

If you want to improve the code for long-term maintenance, consider:

1. Renaming files to English names (see "Recommended new name" column).
2. Moving all hard-coded paths into a `config.yaml` or command-line arguments.
3. Separating `GaugeApp` into smaller modules:
   - `alignment.py`
   - `pointer_segmentation.py`
   - `angle_to_value.py`
   - `visualization.py`
4. Adding unit tests for angle interpolation and IoU calculation.
5. Using `pathlib` consistently instead of mixing `os.path` and `Path`.

These changes are optional for open-sourcing but recommended for maintainability.
