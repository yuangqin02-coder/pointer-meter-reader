import cv2
import numpy as np
import math
import os

# ===============================
# 配置
# ===============================
IMG_PATH = r"1-1_1.jpg"
TEST_IMG_PATH = r"3_3_1_3_1.jpg"
DEBUG_OUTPUT_DIR = r"debug_output"
DISPLAY_SCALE = 0.55
MIN_MASK_PIXELS = 30000
MIN_TIP_DISTANCE_RATIO = 0.55
LOW_LIGHT_MEAN_THRESHOLD = 90
POINTER_ROI_RADIUS_RATIO = 0.86
HIGH_EXPOSURE_ROI_RADIUS_RATIO = 0.6
MIN_ALIGNMENT_IMAGE_SIZE = 400


# ===============================
# 工具函数
# ===============================
def gamma_correct(img, gamma=1.5):
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(img, table)


def enhance_high_exposure(img):
    dark = gamma_correct(img, gamma=1.45)
    lab = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def enhance_low_light(img):
    """
    暗光增强：
    先在 HSV 的亮度通道上做 CLAHE，再用 gamma<1 提亮。
    这和 high_exposure 分支相反，专门用于夜间/欠曝图。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_v = clahe.apply(v_channel)

    table = np.array(
        [((i / 255.0) ** 0.55) * 255 for i in range(256)]
    ).astype("uint8")
    bright_v = cv2.LUT(enhanced_v, table)

    enhanced_hsv = cv2.merge((h_channel, s_channel, bright_v))
    return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)


def upscale_for_alignment(img, min_size=MIN_ALIGNMENT_IMAGE_SIZE):
    h, w = img.shape[:2]
    current_min = min(h, w)
    if current_min >= min_size:
        return img

    scale = min_size / current_min
    return cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )


def red_difference_mask(img):
    b, g, r = cv2.split(img.astype(np.int16))
    red_diff = ((r > 55) & (r > g + 10) & (r > b + 10))
    red_ratio = (r > 80) & (r * 100 > (g + b + 1) * 58)
    return np.where(red_diff | red_ratio, 255, 0).astype(np.uint8)


# 指针分割
def extract_pointer_mask(img, high_exposure=False):
    source_img = enhance_high_exposure(img) if high_exposure else img

    if high_exposure:
        lower1 = np.array([0, 35, 35])
        upper1 = np.array([18, 255, 255])
        lower2 = np.array([150, 35, 35])
    else:
        lower1 = np.array([0, 30, 30])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([145, 35, 35])
    upper2 = np.array([180, 255, 255])

    hsv = cv2.cvtColor(source_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    if high_exposure:
        mask = mask | red_difference_mask(img) | red_difference_mask(source_img)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if high_exposure:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask

# 生成指针掩膜
def generate_pointer_mask(shape, center, angle, radius):
    mask = np.zeros(shape[:2], dtype=np.uint8)

    x = int(center[0] + radius * np.cos(angle))
    y = int(center[1] - radius * np.sin(angle))

    cv2.line(mask, center, (x, y), 255, 6)

    return mask

# IoU计算
def calc_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0


def resize_for_display(img, scale=DISPLAY_SCALE):
    if scale >= 1:
        return img

    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def angle_from_tip(center, tip):
    dx = tip[0] - center[0]
    dy = center[1] - tip[1]
    ang = math.atan2(dy, dx)
    return ang if ang >= 0 else ang + 2 * np.pi


# ===============================
# 主类
# ===============================
class GaugeApp:
    def __init__(self, img_path):
        self.raw_img = cv2.imread(img_path)
        if self.raw_img is None:
            raise Exception("❌ template读取失败")

        self.h, self.w = self.raw_img.shape[:2]

        self.calib_center = None
        self.calib_angles = []


        # 选择 17 个锚点索引与对应值 0.0~1.6 作为插值基准
        self.anchor_indices = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
        self.anchor_values = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4,
             0.5, 0.6, 0.7, 0.8, 0.9,
             1.0, 1.1, 1.2, 1.3, 1.4,
             1.5, 1.6]
        )

        self.radius = int(min(self.h, self.w) * 0.45)
        self.template_dial_circle = self.detect_dial_circle(self.raw_img)

        self.load_calibration()

    # ===============================
    # 读取标定
    # ===============================
    def load_calibration(self):
        if not os.path.exists("calibration_data.txt"):
            raise Exception("❌ 没有 calibration_data.txt")

        angles = []

        with open("calibration_data.txt", "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if parts[0] == "CENTER":
                    self.calib_center = (int(parts[1]), int(parts[2]))
                else:
                    angles.append(float(parts[1]))

        self.calib_angles = np.unwrap(angles)
        print("✅ 标定加载成功")

    def get_anchor_angles(self):
        return np.array([self.calib_angles[i] for i in self.anchor_indices])

    # ===============================
    # 🔥 核心修复：角度归一化
    # ===============================
    def normalize_angle(self, ang):
        base = self.calib_angles[0]

        diff = ang - base
        diff = (diff + np.pi) % (2 * np.pi) - np.pi

        return base + diff

    def angle_to_value(self, ang):
        # 1. 获取锚点角度
        anchor_angles = self.get_anchor_angles()

        # 2. 核心：统一角度周期。将搜索到的角度 ang 映射到锚点的起止范围内
        # 找出锚点的范围
        a_min = min(anchor_angles)
        a_max = max(anchor_angles)

        # 调整 ang 使得它相对于第一个锚点角度处于 [-pi, pi] 之间
        base = anchor_angles[0]
        diff = (ang - base + np.pi) % (2 * np.pi) - np.pi
        target_ang = base + diff

        # 3. 处理 np.interp 要求 xp 递增的问题
        # 如果锚点角度是递减的（顺时针表盘常见情况），我们需要翻转它们进行插值
        if anchor_angles[0] > anchor_angles[-1]:
            # 翻转角度和对应的数值
            value = np.interp(target_ang, anchor_angles[::-1], self.anchor_values[::-1])
        else:
            value = np.interp(target_ang, anchor_angles, self.anchor_values)

        # 4. 打印调试信息（如果你发现还是1，看这里的输出）
        # print(f"DEBUG: target_ang={target_ang:.3f}, range=[{anchor_angles[0]:.3f}, {anchor_angles[-1]:.3f}]")

        return np.clip(value, 0, 1.6)

    def build_static_feature_mask(self, img, center=None, outer_radius=None):
        """
        为图像摆正构建“静态表盘区域”掩膜：
        1. 只保留表盘内部，排除外壳边缘；
        2. 挖掉圆心区域，排除中心轴和指针根部；
        3. 去掉红色区域，避免活动指针和红色字样参与匹配；
        4. 保留表盘内其余区域，让 ORB 自行从黑色刻度线和黑色文字上取稳定特征。
        """
        h, w = img.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        if outer_radius is None:
            outer_radius = int(min(h, w) * 0.46)

        # 先保留内表盘，外壳边缘一律不参与匹配。
        dial_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(dial_mask, center, int(outer_radius * 0.98), 255, -1)

        # 圆心附近是中心轴 + 指针根部，属于动态/高反光区域，直接剔除。
        cv2.circle(dial_mask, center, int(outer_radius * 0.18), 0, -1)

        # 红色区域全部排除：既去掉活动指针，也去掉红色 logo，保证只靠黑色静物对齐。
        red_mask = extract_pointer_mask(img, high_exposure=True)
        red_mask = cv2.dilate(
            red_mask,
            np.ones((9, 9), np.uint8),
            iterations=1,
        )

        static_mask = dial_mask.copy()
        static_mask[red_mask > 0] = 0
        return static_mask

    def detect_dial_circle(self, img):
        """
        检测表盘圆。暗光图里文字特征可能消失，但外圈通常还在，
        可作为 ORB 失败时的几何回退依据。
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if float(gray.mean()) < LOW_LIGHT_MEAN_THRESHOLD:
            gray = cv2.cvtColor(enhance_low_light(img), cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        min_dim = min(img.shape[:2])
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(50, int(min_dim * 0.2)),
            param1=80,
            param2=20,
            minRadius=int(min_dim * 0.25),
            maxRadius=int(min_dim * 0.49),
        )

        if circles is None:
            return None

        # Hough 返回里通常第一个就是最强候选。
        x, y, radius = circles[0][0]
        return float(x), float(y), float(radius)

    def align_by_dial_circle(self, test_img):
        if self.template_dial_circle is None:
            return None

        test_circle = self.detect_dial_circle(test_img)
        if test_circle is None:
            return None

        tx, ty, tr = self.template_dial_circle
        sx, sy, sr = test_circle
        if sr <= 0:
            return None

        scale = tr / sr
        transform = np.float32([
            [scale, 0, tx - scale * sx],
            [0, scale, ty - scale * sy],
        ])
        aligned = cv2.warpAffine(test_img, transform, (self.w, self.h))
        print(
            "✅ 图像摆正回退到表盘圆:"
            f" 源圆心=({sx:.1f}, {sy:.1f}), 半径={sr:.1f}"
        )
        return aligned

    def align_to_template(self, test_img):
        test_img = upscale_for_alignment(test_img)
        fallback = cv2.resize(test_img, (self.w, self.h))

        template_gray = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2GRAY)
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        is_low_light = float(test_gray.mean()) < LOW_LIGHT_MEAN_THRESHOLD

        # 暗图先提亮再做匹配；正常图维持原流程，避免过度增强带来噪点。
        feature_img = enhance_low_light(test_img) if is_low_light else test_img
        feature_gray = cv2.cvtColor(feature_img, cv2.COLOR_BGR2GRAY)

        if is_low_light:
            template_gray = cv2.equalizeHist(template_gray)
            feature_gray = cv2.equalizeHist(feature_gray)

        template_static_mask = self.build_static_feature_mask(
            self.raw_img,
            center=self.calib_center,
            outer_radius=self.radius,
        )
        test_static_mask = self.build_static_feature_mask(feature_img)

        orb = cv2.ORB_create(nfeatures=8000, scaleFactor=1.2, nlevels=8)
        template_kps, template_desc = orb.detectAndCompute(
            template_gray,
            template_static_mask,
        )
        test_kps, test_desc = orb.detectAndCompute(
            feature_gray,
            test_static_mask,
        )

        if template_desc is None or test_desc is None:
            print("⚠️ 图像摆正跳过：未提取到足够特征，回退到 resize。")
            circle_aligned = self.align_by_dial_circle(test_img)
            return circle_aligned if circle_aligned is not None else fallback

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw_matches = matcher.knnMatch(test_desc, template_desc, k=2)

        good_matches = []
        for pair in raw_matches:
            if len(pair) != 2:
                continue
            best, second = pair
            if best.distance < 0.82 * second.distance:
                good_matches.append(best)

        if len(good_matches) < 12:
            print(f"⚠️ 图像摆正跳过：有效匹配点仅 {len(good_matches)} 个，回退到 resize。")
            circle_aligned = self.align_by_dial_circle(test_img)
            return circle_aligned if circle_aligned is not None else fallback

        src_pts = np.float32(
            [test_kps[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [template_kps[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        homography, inlier_mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            8.0,
        )
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

        if homography is None or inliers < 10:
            print(f"⚠️ 图像摆正失败：RANSAC 内点仅 {inliers} 个，回退到 resize。")
            circle_aligned = self.align_by_dial_circle(test_img)
            return circle_aligned if circle_aligned is not None else fallback

        aligned = cv2.warpPerspective(test_img, homography, (self.w, self.h))
        print(
            "✅ 图像摆正成功（静态表盘区域）:"
            f" 匹配点 {len(good_matches)}，RANSAC 内点 {inliers}"
        )
        return aligned

    def detect_angle_by_tip(self, pointer_mask):
        # 选中和圆心附近相连的红色组件，避免红色文字、logo 干扰。
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pointer_mask, 8)
        if num_labels <= 1:
            return None, None, 0

        center_mask = np.zeros_like(pointer_mask)
        cv2.circle(center_mask, self.calib_center, int(self.radius * 0.22), 255, -1)

        candidate_labels = []
        for label in range(1, num_labels):
            component_mask = (labels == label)
            if np.any(component_mask & (center_mask > 0)):
                candidate_labels.append(label)

        if candidate_labels:
            component_mask = np.isin(labels, candidate_labels)
        else:
            selected_label = max(
                range(1, num_labels),
                key=lambda label: stats[label, cv2.CC_STAT_AREA],
            )
            component_mask = (labels == selected_label)

        # 先挖掉中心轴，把真正的细指针和下面较粗的配重尽量分成两段。
        # 再选“径向长度更长、面积相对更小”的那段，避免把配重当作针尖。
        separated_mask = component_mask.astype(np.uint8) * 255
        cv2.circle(separated_mask, self.calib_center, int(self.radius * 0.13), 0, -1)
        sep_labels_count, sep_labels, sep_stats, _ = cv2.connectedComponentsWithStats(
            separated_mask,
            8,
        )

        best_part_mask = None
        best_part_score = -1
        for label in range(1, sep_labels_count):
            area = sep_stats[label, cv2.CC_STAT_AREA]
            if area < 20:
                continue

            part_mask = (sep_labels == label)
            part_ys, part_xs = np.where(part_mask)
            part_dx = part_xs - self.calib_center[0]
            part_dy = part_ys - self.calib_center[1]
            part_dist = np.sqrt(part_dx * part_dx + part_dy * part_dy)
            min_dist = float(np.min(part_dist))
            max_dist = float(np.max(part_dist))
            radial_span = max_dist - min_dist

            if min_dist > self.radius * 0.35 or max_dist < self.radius * 0.25:
                continue

            width_estimate = area / max(radial_span, 1.0)
            score = radial_span * 2.0 + max_dist - width_estimate * 8.0
            if score > best_part_score:
                best_part_score = score
                best_part_mask = part_mask

        if best_part_mask is not None:
            component_mask = best_part_mask

        ys, xs = np.where(component_mask)
        if xs.size == 0:
            return None, None, 0

        dx = xs - self.calib_center[0]
        dy = ys - self.calib_center[1]
        dist = np.sqrt(dx * dx + dy * dy)

        far_count = max(5, int(xs.size * 0.02))
        far_indices = np.argsort(dist)[-far_count:]
        tip_x = int(np.mean(xs[far_indices]))
        tip_y = int(np.mean(ys[far_indices]))
        tip = (tip_x, tip_y)

        best_angle = angle_from_tip(self.calib_center, tip)
        tip_distance = float(np.max(dist))
        return best_angle, tip, tip_distance

    def detect_angle_by_iou(self, real_mask, circle_mask):
        best_score = -1
        best_angle = 0

        for ang in np.linspace(0, 2 * np.pi, 360):
            temp_mask = generate_pointer_mask(
                self.raw_img.shape,
                self.calib_center,
                ang,
                self.radius
            )

            temp_mask = temp_mask & circle_mask
            score = calc_iou(real_mask, temp_mask)

            if score > best_score:
                best_score = score
                best_angle = ang

        return best_angle, best_score

    def build_pointer_mask(self, corrected_img, high_exposure=False):
        real_mask = extract_pointer_mask(corrected_img, high_exposure=high_exposure)

        roi_ratio = HIGH_EXPOSURE_ROI_RADIUS_RATIO if high_exposure else POINTER_ROI_RADIUS_RATIO
        circle_mask = np.zeros_like(real_mask)
        cv2.circle(circle_mask, self.calib_center,
                   int(self.radius * roi_ratio), 255, -1)

        real_mask = real_mask & circle_mask
        return real_mask, circle_mask

    def measure_mask_quality(self, pointer_mask):
        angle, tip, tip_distance = self.detect_angle_by_tip(pointer_mask)
        mask_pixels = int(np.count_nonzero(pointer_mask))
        tip_ratio = tip_distance / self.radius if self.radius > 0 else 0
        return {
            "angle": angle,
            "tip": tip,
            "tip_distance": tip_distance,
            "tip_ratio": tip_ratio,
            "mask_pixels": mask_pixels,
        }

    def should_use_high_exposure_mask(self, quality):
        if quality["angle"] is None:
            return True
        if quality["mask_pixels"] < MIN_MASK_PIXELS:
            return True
        if quality["tip_ratio"] < MIN_TIP_DISTANCE_RATIO:
            return True
        return False

    # ===============================
    # 自动识别
    # ===============================
    def detect(self, test_img):

        corrected_img = self.align_to_template(test_img)
        gray_mean = float(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY).mean())
        low_light_img = None

        if gray_mean < LOW_LIGHT_MEAN_THRESHOLD:
            # 暗光模式下，后续识别直接基于增强后的图像进行，
            # 不再把增强图只当作“补救候选”。
            low_light_img = enhance_low_light(corrected_img)
            real_mask, circle_mask = self.build_pointer_mask(
                low_light_img,
                high_exposure=True,
            )
            quality = self.measure_mask_quality(real_mask)
            mask_mode = "low_light"
            print(
                "🔎 暗光增强分割: "
                f"红色像素={quality['mask_pixels']}, "
                f"针尖距离比例={quality['tip_ratio']:.3f}"
            )
        else:
            real_mask, circle_mask = self.build_pointer_mask(corrected_img, high_exposure=False)
            quality = self.measure_mask_quality(real_mask)
            mask_mode = "normal"

            if self.should_use_high_exposure_mask(quality):
                high_mask, high_circle_mask = self.build_pointer_mask(corrected_img, high_exposure=True)
                high_quality = self.measure_mask_quality(high_mask)
                print(
                    "🔎 普通分割: "
                    f"红色像素={quality['mask_pixels']}, "
                    f"针尖距离比例={quality['tip_ratio']:.3f}"
                )
                print(
                    "🔎 高曝光分割: "
                    f"红色像素={high_quality['mask_pixels']}, "
                    f"针尖距离比例={high_quality['tip_ratio']:.3f}"
                )

                if (
                    quality["angle"] is None
                    or high_quality["tip_ratio"] > quality["tip_ratio"] + 0.08
                    or high_quality["mask_pixels"] > quality["mask_pixels"] * 1.4
                ):
                    real_mask = high_mask
                    circle_mask = high_circle_mask
                    quality = high_quality
                    mask_mode = "high_exposure"

        best_angle = quality["angle"]
        tip = quality["tip"]
        tip_distance = quality["tip_distance"]

        if best_angle is None:
            best_angle, best_score = self.detect_angle_by_iou(real_mask, circle_mask)
            tip = None
            print("⚠️ 针尖检测失败，回退到 IoU 角度匹配。")
        else:
            best_score = tip_distance / self.radius

        value = self.angle_to_value(best_angle)

        # 终端显示结果
        print("\n======================")
        print(f"分割模式: {mask_mode}")
        print(f"红色像素: {quality['mask_pixels']}")
        print(f"角度(rad): {best_angle:.3f}")
        print(f"针尖距离比例: {best_score:.3f}")
        print(f"读数: {value:.3f}")
        print(f"识别输入: {'暗光增强图' if low_light_img is not None else '摆正原图'}")
        print("======================\n")

        return value, best_angle, real_mask, corrected_img, tip, low_light_img

    # ===============================
    # 运行
    # ===============================
    def run(self):
        test_img = cv2.imread(TEST_IMG_PATH)
        if test_img is None:
            raise Exception("❌ 测试图读取失败")

        value, angle, mask, corrected_img, tip, low_light_img = self.detect(test_img)

        # 模板只负责标定和摆正；识别结果画在真正参与识别的摆正图上。
        recognition_img = low_light_img if low_light_img is not None else corrected_img
        result_vis = recognition_img.copy()

        x = int(self.calib_center[0] + self.radius * np.cos(angle))
        y = int(self.calib_center[1] - self.radius * np.sin(angle))

        cv2.line(result_vis, self.calib_center, (x, y), (0, 255, 0), 3)
        cv2.circle(result_vis, self.calib_center, 6, (255, 0, 0), -1)
        if tip is not None:
            cv2.circle(result_vis, tip, 8, (0, 255, 255), -1)
            cv2.line(result_vis, self.calib_center, tip, (0, 255, 0), 3)

        cv2.putText(result_vis, f"Value: {value:.3f}",
                    (20, 50), 1, 2, (0, 0, 255), 2)

        os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
        original_path = os.path.join(DEBUG_OUTPUT_DIR, "01_original.jpg")
        aligned_path = os.path.join(DEBUG_OUTPUT_DIR, "02_aligned_to_template.jpg")
        result_path = os.path.join(DEBUG_OUTPUT_DIR, "03_result_on_aligned.jpg")
        mask_path = os.path.join(DEBUG_OUTPUT_DIR, "04_pointer_mask.jpg")
        low_light_path = os.path.join(DEBUG_OUTPUT_DIR, "06_low_light_enhanced.jpg")

        cv2.imwrite(original_path, test_img)
        cv2.imwrite(aligned_path, corrected_img)
        cv2.imwrite(result_path, result_vis)
        cv2.imwrite(mask_path, mask)
        if low_light_img is not None:
            cv2.imwrite(low_light_path, low_light_img)

        print(f"✅ 原始图片已保存: {original_path}")
        print(f"✅ 摆正后的图片已保存: {aligned_path}")
        print(f"✅ 识别结果画在摆正图上已保存: {result_path}")
        print(f"✅ 指针Mask已保存: {mask_path}")
        if low_light_img is not None:
            print(f"✅ 暗光增强图已保存: {low_light_path}")

        cv2.imshow("01 Original Image", resize_for_display(test_img))
        cv2.imshow("02 Aligned To Template", resize_for_display(corrected_img))
        cv2.imshow("03 Result On Aligned", resize_for_display(result_vis))
        if low_light_img is not None:
            cv2.imshow("06 Low Light Enhanced", resize_for_display(low_light_img))
        cv2.imshow("Mask", resize_for_display(mask))

        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ===============================
# 入口
# ===============================
if __name__ == "__main__":
    app = GaugeApp(IMG_PATH)
    app.run()

