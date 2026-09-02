import cv2
import numpy as np

from read_copy_bai_new import GaugeApp


class EqualAngleLinearReader(GaugeApp):
    """Proposed angle detector with endpoint-based equal-angle mapping."""

    method_name = "equal_angle_linear"

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
        value, angle, mask, corrected, tip, low_light = super().detect(test_img)
        recognition_img = low_light if low_light is not None else corrected

        angle_vis = recognition_img.copy()
        end_x = int(self.calib_center[0] + self.radius * np.cos(angle))
        end_y = int(self.calib_center[1] - self.radius * np.sin(angle))
        cv2.line(angle_vis, self.calib_center, (end_x, end_y), (0, 255, 0), 3)
        cv2.circle(angle_vis, self.calib_center, 6, (255, 0, 0), -1)
        if tip is not None:
            cv2.circle(angle_vis, tip, 7, (0, 255, 255), -1)

        mapping_vis = angle_vis.copy()
        cv2.putText(
            mapping_vis,
            f"linear: angle={angle:.3f}, value={value:.3f}",
            (12, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 255),
            2,
        )
        self.last_debug_images = {
            "01_input": test_img,
            "02_orb_aligned": corrected,
            "03_recognition_input": recognition_img,
            "04_pointer_mask": mask,
            "05_tip_and_angle": angle_vis,
            "06_equal_angle_mapping": mapping_vis,
        }
        return value, angle, mask, corrected, tip, low_light
