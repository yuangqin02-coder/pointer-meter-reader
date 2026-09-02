import argparse
import csv
import math
import os
from pathlib import Path

import cv2

from read_copy_bai_new import GaugeApp


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE_PATH = SCRIPT_DIR / "1-1_1.jpg"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "batch_image_results"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量处理文件夹中的仪表图片，并按图片名分别保存识别结果。"
    )
    parser.add_argument(
        "image_dir",
        help="待处理图片文件夹",
    )
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE_PATH),
        help="模板图片路径，默认使用脚本目录下的 1-1_1.jpg",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="总输出文件夹，默认保存到 batch_image_results",
    )
    parser.add_argument(
        "--include-template",
        action="store_true",
        help="默认跳过模板图；加上这个参数后也处理模板图",
    )
    return parser.parse_args()


def collect_images(image_dir, template_path, include_template):
    image_dir = Path(image_dir)
    template_name = Path(template_path).name
    images = []

    for path in sorted(image_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        if not include_template and path.name == template_name:
            continue
        images.append(path)

    return images


def draw_result(app, value, angle, corrected_img, tip, low_light_img):
    recognition_img = low_light_img if low_light_img is not None else corrected_img
    result_vis = recognition_img.copy()

    x = int(app.calib_center[0] + app.radius * math.cos(angle))
    y = int(app.calib_center[1] - app.radius * math.sin(angle))

    cv2.line(result_vis, app.calib_center, (x, y), (0, 255, 0), 3)
    cv2.circle(result_vis, app.calib_center, 6, (255, 0, 0), -1)
    if tip is not None:
        cv2.circle(result_vis, tip, 8, (0, 255, 255), -1)
        cv2.line(result_vis, app.calib_center, tip, (0, 255, 0), 3)

    cv2.putText(
        result_vis,
        f"Value: {value:.3f}",
        (20, 50),
        1,
        2,
        (0, 0, 255),
        2,
    )
    return result_vis


def save_image_results(output_root, image_path, original_img, corrected_img, result_img, mask, low_light_img):
    out_dir = Path(output_root) / Path(image_path).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "01_original.jpg"), original_img)
    cv2.imwrite(str(out_dir / "02_aligned_to_template.jpg"), corrected_img)
    cv2.imwrite(str(out_dir / "03_result_on_aligned.jpg"), result_img)
    cv2.imwrite(str(out_dir / "04_pointer_mask.jpg"), mask)
    if low_light_img is not None:
        cv2.imwrite(str(out_dir / "05_low_light_enhanced.jpg"), low_light_img)

    return out_dir


def write_summary(output_root, rows):
    summary_path = Path(output_root) / "summary.csv"
    fieldnames = ["image", "value", "angle_rad", "success", "output_dir", "message"]

    with open(summary_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return summary_path


def main():
    args = parse_args()

    image_dir = Path(args.image_dir).expanduser().resolve()
    template_path = Path(args.template).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not image_dir.is_dir():
        raise RuntimeError(f"待处理文件夹不存在: {image_dir}")
    if not template_path.is_file():
        raise RuntimeError(f"模板图片不存在: {template_path}")

    # read_copy_bai_new.py 里按相对路径读取 calibration_data.txt，
    # 切到脚本目录后，批处理和单张运行使用同一份标定文件。
    os.chdir(SCRIPT_DIR)

    images = collect_images(image_dir, template_path, args.include_template)
    if not images:
        raise RuntimeError(f"没有找到可处理图片: {image_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    app = GaugeApp(str(template_path))
    rows = []

    print(f"待处理图片数量: {len(images)}")
    print(f"图片文件夹: {image_dir}")
    print(f"总输出文件夹: {output_root}")

    for index, image_path in enumerate(images, start=1):
        print(f"\n[{index}/{len(images)}] 处理: {image_path.name}")
        row = {
            "image": image_path.name,
            "value": "",
            "angle_rad": "",
            "success": "no",
            "output_dir": "",
            "message": "",
        }

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise RuntimeError("图片读取失败")

            value, angle, mask, corrected_img, tip, low_light_img = app.detect(img)
            result_img = draw_result(
                app,
                float(value),
                float(angle),
                corrected_img,
                tip,
                low_light_img,
            )
            out_dir = save_image_results(
                output_root,
                image_path,
                img,
                corrected_img,
                result_img,
                mask,
                low_light_img,
            )

            row["value"] = f"{float(value):.6f}"
            row["angle_rad"] = f"{float(angle):.6f}"
            row["success"] = "yes"
            row["output_dir"] = str(out_dir)
            print(f"✅ 已保存: {out_dir}")
        except Exception as e:
            row["message"] = str(e)
            print(f"❌ 处理失败: {e}")

        rows.append(row)

    summary_path = write_summary(output_root, rows)
    success_count = sum(1 for row in rows if row["success"] == "yes")

    print("\n======================")
    print(f"批量处理完成: {success_count}/{len(rows)}")
    print(f"总输出文件夹: {output_root}")
    print(f"汇总表: {summary_path}")
    print("======================")


if __name__ == "__main__":
    main()

