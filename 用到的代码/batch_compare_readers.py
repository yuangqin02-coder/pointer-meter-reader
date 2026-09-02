import argparse
import csv
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np

from equal_angle_linear_reader import EqualAngleLinearReader
from read_copy_bai_new import GaugeApp
from traditional_hough_reader import TraditionalHoughReader


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = SCRIPT_DIR / "1-1_1.jpg"
DEFAULT_OUTPUT = SCRIPT_DIR / "reader_comparison_results"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two baselines with the proposed reader.")
    parser.add_argument("image_dir", help="Directory containing cropped gauge images")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--ground-truth",
        help=(
            "Optional CSV/TSV containing image and true_value or human_value. "
            "If omitted, image_dir/summary.csv is used when present."
        ),
    )
    parser.add_argument("--include-template", action="store_true")
    return parser.parse_args()


def load_ground_truth(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
        file.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        rows = csv.DictReader(file, dialect=dialect)
        fields = set(rows.fieldnames or [])
        value_field = "true_value" if "true_value" in fields else "human_value"
        if "image" not in fields or value_field not in fields:
            raise RuntimeError(
                "Ground-truth table must contain image and true_value or human_value"
            )
        return {
            row["image"]: float(row[value_field])
            for row in rows
            if row.get("image") and row.get(value_field)
        }


def collect_images(image_dir, template_path, include_template):
    image_dir = Path(image_dir).resolve()
    template_path = Path(template_path).resolve()
    direct_images = [
        (path.name, path)
        for path in sorted(image_dir.iterdir())
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTS
        and (include_template or path.resolve() != template_path)
    ]
    if direct_images:
        return direct_images

    # Supports batch_read_copy_bai_new.py output:
    # sample_name/01_original.jpg
    nested_images = []
    for sample_dir in sorted(image_dir.iterdir()):
        original = sample_dir / "01_original.jpg"
        if sample_dir.is_dir() and original.is_file():
            nested_images.append((f"{sample_dir.name}.jpg", original))
    return nested_images


def draw_result(reader, value, angle, corrected_img, tip):
    result = corrected_img.copy()
    x = int(reader.calib_center[0] + reader.radius * math.cos(angle))
    y = int(reader.calib_center[1] - reader.radius * math.sin(angle))
    cv2.line(result, reader.calib_center, (x, y), (0, 255, 0), 3)
    cv2.circle(result, reader.calib_center, 5, (255, 0, 0), -1)
    if tip is not None:
        cv2.circle(result, tip, 7, (0, 255, 255), -1)
    cv2.putText(result, f"{value:.3f}", (15, 40), 1, 2, (0, 0, 255), 2)
    return result


def save_debug_images(result_dir, reader):
    for step_name, image in getattr(reader, "last_debug_images", {}).items():
        cv2.imwrite(str(result_dir / f"{step_name}.jpg"), image)


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def calculate_metrics(rows, total_images):
    metrics = []
    methods = sorted({row["method"] for row in rows})
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        successful = [row for row in method_rows if row["success"] == "yes"]
        errors = [
            float(row["abs_error"])
            for row in successful
            if row["abs_error"] != ""
        ]
        times = [float(row["time_ms"]) for row in successful]
        metrics.append(
            {
                "method": method,
                "success_rate": len(successful) / total_images if total_images else 0,
                "mae": float(np.mean(errors)) if errors else "",
                "rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else "",
                "max_error": max(errors) if errors else "",
                "accuracy_within_0.05": (
                    sum(error <= 0.05 for error in errors) / len(errors) if errors else ""
                ),
                "avg_time_ms": float(np.mean(times)) if times else "",
            }
        )
    return metrics


def main():
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    template = Path(args.template).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not image_dir.is_dir():
        raise RuntimeError(f"Image directory does not exist: {image_dir}")
    if not template.is_file():
        raise RuntimeError(f"Template does not exist: {template}")

    os.chdir(SCRIPT_DIR)
    ground_truth_path = args.ground_truth
    if not ground_truth_path and (image_dir / "summary.csv").is_file():
        ground_truth_path = str(image_dir / "summary.csv")
        print(f"Auto ground truth: {ground_truth_path}")
    truth = load_ground_truth(ground_truth_path)
    images = collect_images(image_dir, template, args.include_template)
    if not images:
        raise RuntimeError(f"No images found in: {image_dir}")

    readers = {
        "traditional_hough": TraditionalHoughReader(str(template)),
        "equal_angle_linear": EqualAngleLinearReader(str(template)),
        "proposed_multi_anchor": GaugeApp(str(template)),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for image_index, (sample_name, image_path) in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[{image_index}/{len(images)}] Cannot read {sample_name}")
            continue

        for method, reader in readers.items():
            row = {
                "image": sample_name,
                "method": method,
                "true_value": truth.get(sample_name, ""),
                "predicted_value": "",
                "abs_error": "",
                "angle_rad": "",
                "time_ms": "",
                "success": "no",
                "message": "",
            }
            try:
                started = time.perf_counter()
                value, angle, mask, corrected, tip, low_light = reader.detect(image)
                elapsed_ms = (time.perf_counter() - started) * 1000

                result_dir = output_root / method / Path(sample_name).stem
                result_dir.mkdir(parents=True, exist_ok=True)
                recognition_image = low_light if low_light is not None else corrected
                result = draw_result(reader, float(value), float(angle), recognition_image, tip)
                cv2.imwrite(str(result_dir / "aligned.jpg"), corrected)
                cv2.imwrite(str(result_dir / "mask.jpg"), mask)
                cv2.imwrite(str(result_dir / "result.jpg"), result)
                save_debug_images(result_dir, reader)

                row["predicted_value"] = f"{float(value):.6f}"
                row["angle_rad"] = f"{float(angle):.6f}"
                row["time_ms"] = f"{elapsed_ms:.3f}"
                row["success"] = "yes"
                if row["true_value"] != "":
                    row["abs_error"] = f"{abs(float(value) - float(row['true_value'])):.6f}"
            except Exception as error:
                row["message"] = str(error)
            rows.append(row)
            print(
                f"[{image_index}/{len(images)}] {sample_name} | "
                f"{method} | {row['success']} | {row['predicted_value']}"
            )

    summary_fields = [
        "image",
        "method",
        "true_value",
        "predicted_value",
        "abs_error",
        "angle_rad",
        "time_ms",
        "success",
        "message",
    ]
    write_csv(output_root / "summary.csv", rows, summary_fields)

    metrics = calculate_metrics(rows, len(images))
    metric_fields = [
        "method",
        "success_rate",
        "mae",
        "rmse",
        "max_error",
        "accuracy_within_0.05",
        "avg_time_ms",
    ]
    write_csv(output_root / "metrics.csv", metrics, metric_fields)
    print(f"Summary: {output_root / 'summary.csv'}")
    print(f"Metrics: {output_root / 'metrics.csv'}")


if __name__ == "__main__":
    main()
