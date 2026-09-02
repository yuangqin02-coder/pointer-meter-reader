import cv2
import numpy as np
import math
import os

# ==========================================
# 配置区域
# ==========================================
IMG_PATH = r"1-1_1.jpg"


# ==========================================

class GaugeApp:
    def __init__(self, img_path):
        self.img_path = img_path
        self.raw_img = cv2.imread(img_path)

        if self.raw_img is None:
            print(f"❌ 错误：无法读取图片: {img_path}")
            exit()

        self.h, self.w = self.raw_img.shape[:2]
        self.calib_center = None
        self.calib_angles = []

        # 定义锚点
        self.anchor_indices = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
        self.anchor_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])

        self.is_calibrating = False
        self.current_value = 0.0
        self.is_dragging = False
        self.radius = int(min(self.h, self.w) * 0.45)

        # 启动即读取文件
        self.load_existing_calibration()

    def load_existing_calibration(self):
        file_name = "calibration_data.txt"
        if os.path.exists(file_name):
            angles = []
            try:
                with open(file_name, "r") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if parts[0] == "CENTER":
                            self.calib_center = (int(parts[1]), int(parts[2]))
                        else:
                            angles.append(float(parts[1]))

                # 【核心修复】使用 unwrap 确保角度序列是单调递增/递减的，解决 0/360度突变问题
                self.calib_angles = np.unwrap(angles).tolist()
                print(f"📂 已加载标定文件，处理了角度连续性。")
            except Exception as e:
                print(f"⚠️ 读取文件失败: {e}")
        else:
            print("💡 未发现标定文件，请按 'c' 键开始标定。")

    def get_anchor_angles(self):
        return np.array([self.calib_angles[i] for i in self.anchor_indices])

    def value_to_angle(self, val):
        if not self.calib_angles: return 0
        return np.interp(val, self.anchor_values, self.get_anchor_angles())

    def angle_to_value(self, ang):
        if not self.calib_angles: return 0
        anchor_angles = self.get_anchor_angles()

        # 将输入角度映射到标定角度的展开范围内
        # 找到与标定起点最接近的周期
        base_angle = anchor_angles[0]
        diff = ang - base_angle
        # 修正到 -pi 到 pi 范围内
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        corrected_ang = base_angle + diff

        return np.interp(corrected_ang, anchor_angles, self.anchor_values)

    def save_calibration(self):
        raw_angles = []
        for p in self.calib_points:
            dx = p[0] - self.calib_center[0]
            dy = self.calib_center[1] - p[1]  # 抵消Y轴翻转
            angle = math.atan2(dy, dx)
            raw_angles.append(angle)

        # 再次使用 unwrap 保证保存的数据是连续的
        self.calib_angles = np.unwrap(raw_angles).tolist()

        with open("calibration_data.txt", "w") as f:
            f.write(f"CENTER,{self.calib_center[0]},{self.calib_center[1]}\n")
            for i in range(32):
                v = 1.6 * i / 31
                f.write(f"{v:.4f},{self.calib_angles[i]:.6f}\n")

        print("✅ 标定成功，已处理跨零点逻辑。")
        self.is_calibrating = False

    def mouse_callback(self, event, x, y, flags, param):
        if self.is_calibrating:
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.calib_center is None:
                    self.calib_center = (x, y)
                elif len(self.calib_points) < 32:
                    self.calib_points.append((x, y))
                    if len(self.calib_points) == 32:
                        self.save_calibration()
            return

        if self.calib_center and self.calib_angles:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.is_dragging = True
            elif event == cv2.EVENT_LBUTTONUP:
                self.is_dragging = False

            if self.is_dragging or event == cv2.EVENT_LBUTTONDOWN:
                dx = x - self.calib_center[0]
                dy = self.calib_center[1] - y
                ang = math.atan2(dy, dx)
                self.current_value = self.angle_to_value(ang)
                cv2.setTrackbarPos("Value", "Gauge", int(self.current_value * 100))

    def run(self):
        cv2.namedWindow("Gauge", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Gauge", self.mouse_callback)

        def on_trackbar(v):
            if not self.is_calibrating:
                self.current_value = v / 100.0

        cv2.createTrackbar("Value", "Gauge", 0, 160, on_trackbar)

        while True:
            vis = self.raw_img.copy()

            if self.is_calibrating:
                if self.calib_center:
                    cv2.circle(vis, self.calib_center, 5, (0, 0, 255), -1)
                for i, p in enumerate(self.calib_points):
                    cv2.circle(vis, p, 3, (0, 255, 0), -1)
                cv2.putText(vis, f"Point: {len(self.calib_points)}/32", (20, 40), 1, 2, (0, 0, 255), 2)

            elif self.calib_center and len(self.calib_angles) == 32:
                # 绘制当前指针
                draw_ang = self.value_to_angle(self.current_value)
                tx = int(self.calib_center[0] + self.radius * math.cos(draw_ang))
                ty = int(self.calib_center[1] - self.radius * math.sin(draw_ang))

                cv2.line(vis, self.calib_center, (tx, ty), (0, 0, 255), 3)
                cv2.circle(vis, self.calib_center, 8, (0, 0, 255), -1)
                cv2.putText(vis, f"Value: {self.current_value:.3f}", (20, 50), 1, 2, (0, 0, 255), 2)

            cv2.imshow("Gauge", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            if key == ord('c'):
                self.is_calibrating = True
                self.calib_center = None
                self.calib_points = []

        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = GaugeApp(IMG_PATH)
    app.run()
