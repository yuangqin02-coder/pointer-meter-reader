import cv2
import numpy as np

from read_copy_bai_new import (
    GaugeApp,
    POINTER_ROI_RADIUS_RATIO,
    angle_from_tip,
)


class TraditionalHoughReader(GaugeApp):
    """Shared ORB alignment + HSV segmentation + Hough line + linear mapping."""

    method_name = "traditional_hough"

    def angle_to_value(self, angle):
        start_angle = float(self.calib_angles[0])
        end_angle = float(self.calib_angles[-1])
        direction = 1.0 if end_angle >= start_angle else -1.0
        span = abs(end_angle - start_angle)

        target = start_angle + ((angle - start_angle + np.pi) % (2 * np.pi) - np.pi)
        while direction * (target - start_angle) < 0:
            target += direction * 2 * np.pi
        while direction * (target - end_angle) > 0:
            target -= direction * 2 * np.pi

        ratio = direction * (target - start_angle) / span
        return float(np.clip(ratio * 1.6, 0.0, 1.6))

    def detect(self, test_img):
        corrected_img = self.align_to_template(test_img)

        # Traditional fixed HSV threshold segmentation.
        hsv = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2HSV)
        raw_mask = (
            cv2.inRange(hsv, np.array([0, 30, 30]), np.array([10, 255, 255]))
            | cv2.inRange(hsv, np.array([145, 35, 35]), np.array([180, 255, 255]))
        )
        kernel = np.ones((3, 3), np.uint8)
        pointer_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)

        roi_mask = np.zeros_like(pointer_mask)
        cv2.circle(
            roi_mask,
            self.calib_center,
            int(self.radius * POINTER_ROI_RADIUS_RATIO),
            255,
            -1,
        )
        pointer_mask &= roi_mask

        edges = cv2.Canny(pointer_mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=25,
            minLineLength=max(20, int(self.radius * 0.25)),
            maxLineGap=max(5, int(self.radius * 0.12)),
        )
        if lines is None:
            raise RuntimeError("Hough method did not detect a pointer line")

        cx, cy = self.calib_center
        all_lines_vis = corrected_img.copy()
        best = None
        best_score = -np.inf
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(all_lines_vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
            segment = np.array([x2 - x1, y2 - y1], dtype=np.float64)
            length = float(np.linalg.norm(segment))
            if length == 0:
                continue

            center_offset = np.array([cx - x1, cy - y1], dtype=np.float64)
            center_distance = abs(float(np.cross(segment, center_offset))) / length
            d1 = float(np.hypot(x1 - cx, y1 - cy))
            d2 = float(np.hypot(x2 - cx, y2 - cy))
            far_distance = max(d1, d2)

            if center_distance > self.radius * 0.15:
                continue
            if far_distance < self.radius * 0.30:
                continue

            score = length + 0.35 * far_distance - 2.0 * center_distance
            if score > best_score:
                best_score = score
                best = (x1, y1, x2, y2, d1, d2)

        if best is None:
            raise RuntimeError("Hough lines were found, but none passed the center constraint")

        x1, y1, x2, y2, d1, d2 = best
        tip = (int(x1), int(y1)) if d1 >= d2 else (int(x2), int(y2))
        angle = angle_from_tip(self.calib_center, tip)
        value = self.angle_to_value(angle)

        selected_line_vis = corrected_img.copy()
        cv2.line(selected_line_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(selected_line_vis, self.calib_center, 6, (255, 0, 0), -1)
        cv2.circle(selected_line_vis, tip, 7, (0, 255, 255), -1)
        cv2.putText(
            selected_line_vis,
            f"angle={angle:.3f}, value={value:.3f}",
            (12, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )

        self.last_debug_images = {
            "01_input": test_img,
            "02_orb_aligned": corrected_img,
            "03_hsv_raw_mask": raw_mask,
            "04_morphology_mask": pointer_mask,
            "05_canny_edges": edges,
            "06_hough_candidates": all_lines_vis,
            "07_selected_line_and_mapping": selected_line_vis,
        }
        return value, angle, pointer_mask, corrected_img, tip, None
