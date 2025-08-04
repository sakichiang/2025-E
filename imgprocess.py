# imgprocess.py
# 图像处理模块 - 专门用来检测A4纸等矩形目标
# 这个模块能够：
# 1. 从摄像头或图片中检测白色矩形目标（比如A4纸）
# 2. 自动调节曝光获得更好的检测效果
# 3. 处理不完整的目标（比如纸张被遮挡或超出画面）
# 4. 模拟激光点位置用于瞄准

import cv2
import numpy as np
import sys
from collections import deque
import math

# ==================== 所有配置都在这里，方便调整 ====================

# 运行模式选择
RUN_MODE = 'camera'  # 'camera' 表示用摄像头，'image' 表示处理单张图片

# 摄像头设置
SET_RESOLUTION = True    # 是否手动设置分辨率
FRAME_WIDTH = 640        # 图像宽度
FRAME_HEIGHT = 480       # 图像高度

# 自适应曝光设置 - 让程序自动调节亮度获得更好的检测效果
ADAPTIVE_EXPOSURE = True        # 是否开启自适应曝光
TARGET_BRIGHTNESS = 90          # 希望图像的平均亮度（0-255）
BRIGHTNESS_TOLERANCE = 5        # 亮度误差容忍度
ADAPTIVE_GAIN = 0.1             # 曝光调整的速度，越小越平滑
EXPOSURE_MIN = 20               # 最小曝光值
EXPOSURE_MAX = 400              # 最大曝光值
BRIGHTNESS_HISTORY_LEN = 5      # 用几帧来平滑亮度

# 传统曝光设置（仅在关闭自适应曝光时生效）
AUTO_EXPOSURE = False           # 是否用摄像头自动曝光
MANUAL_EXPOSURE_VALUE = 150     # 手动曝光值

# 目标检测参数
A4_ASPECT_RATIO = 1.414         # A4纸的长宽比（√2）
ASPECT_RATIO_TOLERANCE = 0.25   # 长宽比的容忍度，透视会改变比例

# 图像预处理
CROP_TOP_PIXELS = 10            # 从顶部裁掉多少像素
CROP_BOTTOM_PIXELS = 10         # 从底部裁掉多少像素
MIN_CONTOUR_BRIGHTNESS = 120    # 有效目标的最小亮度（很重要！）
MIN_CONTOUR_AREA = 1000         # 轮廓的最小面积
RECONSTRUCTION_AREA_RATIO_LIMIT = 5.0  # 重建面积限制，防止误判

# 相机标定（可选）
USE_CALIBRATION = True          # 是否使用相机标定去畸变
CAMERA_MATRIX = None
DIST_COEFFS = None

# 调试设置
HEADLESS_MODE = True            # 无头模式，不显示窗口
IMAGE_PATH_FOR_DEBUG = 'target2.jpg'  # 调试用的图片路径

# 模拟激光点设置
SIMULATE_LASER_DOT = True       # 是否显示模拟的激光点
LASER_OFFSET_X_PIXELS = -20     # 激光点相对于图像中心的水平偏移
LASER_OFFSET_Y_PIXELS = -15     # 激光点相对于图像中心的垂直偏移
LASER_DOT_COLOR_BGR = (0, 255, 0)  # 激光点颜色（绿色）

# 边界检测算法（当目标不完整时的备用方案）
ENABLE_BORDER_DETECTION = False    # 是否启用边界检测
FIXED_POINT_OFFSET = 50            # 边界模式下的固定偏移
BORDER_CHECK_STRIP_WIDTH = 30      # 检查边缘的宽度
LR_BORDER_CONTOUR_MIN_AREA = 50    # 左右边缘轮廓最小面积
TB_BORDER_CONTOUR_MIN_AREA = 100   # 上下边缘轮廓最小面积
BORDER_LINE_ORIENTATION_RATIO = 3  # 线条的长宽比阈值

# =================================================================

# 加载相机标定文件
if USE_CALIBRATION:
    try:
        with np.load('camera_calibration.npz') as data:
            CAMERA_MATRIX = data['camera_matrix']
            DIST_COEFFS = data['dist_coeffs']
            print("成功加载相机标定文件")
    except FileNotFoundError:
        print("警告: 没找到标定文件 'camera_calibration.npz'")
        print("如果需要标定，请先运行 calibration.py")
        USE_CALIBRATION = False

def resize_for_display(image, max_width=1600, max_height=900):
    """
    把图像缩放到合适的显示尺寸
    主要是为了防止图像太大超出屏幕
    """
    img_h, img_w = image.shape[:2]
    if img_w <= max_width and img_h <= max_height:
        return image

    scale = min(max_width / img_w, max_height / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def line_intersection(line1, line2):
    """
    计算两条线的交点
    用来找矩形对角线的交点，也就是中心点
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None  # 平行线，没有交点

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)

def _reconstruct_quad_from_partial_contour(contour, aspect_ratio, image_shape):
    """
    从部分轮廓重建完整的四边形
    比如当A4纸只有一部分在画面中时，尝试推测完整的形状
    """
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) < 2:
        return None

    # 找到最长的边
    max_dist = -1
    p1_idx, p2_idx = -1, -1
    for i in range(len(approx)):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % len(approx)][0]
        dist = np.linalg.norm(p1 - p2)
        if dist > max_dist:
            max_dist = dist
            p1_idx, p2_idx = i, (i + 1) % len(approx)

    if max_dist < 50:  # 边太短就不处理了
        return None

    # 根据长宽比推测另一边的长度
    p1 = approx[p1_idx][0]
    p2 = approx[p2_idx][0]
    vec = p2 - p1
    perp_vec = np.array([-vec[1], vec[0]])  # 垂直向量
    perp_vec_norm = perp_vec / np.linalg.norm(perp_vec)

    other_side_len1 = max_dist / aspect_ratio
    other_side_len2 = max_dist * aspect_ratio
    other_side_len = min(other_side_len1, other_side_len2)

    # 构造完整的四边形
    p3 = p2 + perp_vec_norm * other_side_len
    p4 = p1 + perp_vec_norm * other_side_len
    reconstructed_points = np.array([p1, p2, p3, p4], dtype=np.int32)

    # 检查重建的面积是否合理
    partial_area = cv2.contourArea(contour)
    reconstructed_area = cv2.contourArea(reconstructed_points)
    image_area = image_shape[0] * image_shape[1]

    # 一些基本的合理性检查
    if reconstructed_area <= partial_area:
        return None

    if partial_area > 0 and (reconstructed_area / partial_area) > RECONSTRUCTION_AREA_RATIO_LIMIT:
        return None

    if reconstructed_area > image_area * 0.95:
        return None

    return reconstructed_points

class FrameDetector:
    """
    主要的检测类
    功能：
    1. 初始化摄像头
    2. 检测图像中的矩形目标
    3. 自动调节曝光
    4. 返回目标中心和激光点位置
    """
    def __init__(self, use_camera=True, max_cameras_to_check=5):
        self.cap = None
        self.target_exposure = float(MANUAL_EXPOSURE_VALUE)
        self.last_set_exposure = self.target_exposure
        self.brightness_history = deque(maxlen=BRIGHTNESS_HISTORY_LEN)
        self.mean_brightness = 0

        if use_camera:
            # 尝试找到可用的摄像头
            found_camera = False
            for i in range(max_cameras_to_check):
                cap_test = cv2.VideoCapture(i)
                if cap_test is not None and cap_test.isOpened():
                    # 设置分辨率
                    if SET_RESOLUTION:
                        cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                        cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

                    # 设置曝光
                    if ADAPTIVE_EXPOSURE or not AUTO_EXPOSURE:
                        cap_test.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手动曝光
                        cap_test.set(cv2.CAP_PROP_EXPOSURE, self.target_exposure)
                    else:
                        cap_test.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 自动曝光

                    self.cap = cap_test
                    w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"找到摄像头 {i}，分辨率: {w}x{h}")
                    found_camera = True
                    break
                else:
                    cap_test.release()

            if not found_camera:
                raise IOError(f"没找到可用的摄像头（尝试了 0-{max_cameras_to_check-1}）")

    def _adjust_exposure(self, gray_frame, roi_contour=None):
        """
        根据图像亮度自动调节曝光
        如果检测到目标，就基于目标区域调节；否则基于图像中心区域
        """
        if not ADAPTIVE_EXPOSURE:
            return

        # 计算当前亮度
        if roi_contour is not None and cv2.contourArea(roi_contour) > 100:
            # 基于检测到的目标区域
            mask = np.zeros(gray_frame.shape, dtype=np.uint8)
            cv2.drawContours(mask, [roi_contour], -1, 255, -1)
            current_brightness = cv2.mean(gray_frame, mask=mask)[0]
        else:
            # 基于图像中心区域
            h, w = gray_frame.shape
            center_rect = gray_frame[h//4:h*3//4, w//4:w*3//4]
            current_brightness = np.mean(center_rect)

        # 用历史亮度做平滑
        self.brightness_history.append(current_brightness)
        self.mean_brightness = np.mean(self.brightness_history)

        # 计算需要调节的量
        error = TARGET_BRIGHTNESS - self.mean_brightness
        if abs(error) > BRIGHTNESS_TOLERANCE:
            self.target_exposure += error * ADAPTIVE_GAIN
            self.target_exposure = max(EXPOSURE_MIN, min(self.target_exposure, EXPOSURE_MAX))

            # 只有变化比较大时才实际设置，避免频繁调节
            if abs(int(self.target_exposure) - self.last_set_exposure) > 1:
                self.last_set_exposure = int(self.target_exposure)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.last_set_exposure)

    def _process_frame(self, frame):
        """
        处理单帧图像的核心函数
        返回：目标中心、激光点位置、处理后的图像、灰度图、二值图
        """
        # 裁剪图像顶部和底部（如果设置了的话）
        if CROP_TOP_PIXELS > 0 or CROP_BOTTOM_PIXELS > 0:
            h_orig = frame.shape[0]
            crop_top = CROP_TOP_PIXELS
            crop_bottom = h_orig - CROP_BOTTOM_PIXELS
            if crop_bottom > crop_top:
                frame = frame[crop_top:crop_bottom, :]
            else:
                print("警告: 裁剪设置有问题，跳过裁剪")

        # 相机标定去畸变（如果有标定文件的话）
        if USE_CALIBRATION and CAMERA_MATRIX is not None and DIST_COEFFS is not None:
            frame = cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFFS, None, CAMERA_MATRIX)

        # 转灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 调节曝光
        if self.cap and self.cap.isOpened():
            self._adjust_exposure(gray)

        # 图像处理：高斯模糊 + 自适应二值化
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 15, 4)

        # 找轮廓
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 筛选有效轮廓
        valid_contours = []
        if contours and hierarchy is not None:
            for i, contour in enumerate(contours):
                # 跳过内部轮廓（只要外轮廓）
                if hierarchy[0][i][3] != -1:
                    continue

                # 面积太小的跳过
                if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                    continue

                # 检查长宽比
                rect = cv2.minAreaRect(contour)
                (w_rect, h_rect) = rect[1]
                if w_rect == 0 or h_rect == 0:
                    continue
                ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
                if abs(ratio - A4_ASPECT_RATIO) > ASPECT_RATIO_TOLERANCE:
                    continue

                # 检查亮度（这个很重要！）
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_brightness = cv2.mean(gray, mask=mask)[0]
                if mean_brightness < MIN_CONTOUR_BRIGHTNESS:
                    continue

                valid_contours.append(contour)

        # 选择最大的有效轮廓作为目标
        target_contour = None
        if valid_contours:
            target_contour = max(valid_contours, key=cv2.contourArea)

        # 再次调节曝光（基于检测到的目标）
        if self.cap and self.cap.isOpened() and target_contour is not None:
            self._adjust_exposure(gray, roi_contour=target_contour)

        # 计算目标中心
        frame_center = None
        h, w = frame.shape[:2]
        is_complete_quad = False

        if target_contour is not None:
            # 检查目标是否完整（没有被画面边缘裁切）
            border_margin = 5
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(target_contour)
            is_clipped = (x_rect <= border_margin or y_rect <= border_margin or
                          x_rect + w_rect >= w - border_margin or y_rect + h_rect >= h - border_margin)

            # 尝试用多边形拟合
            perimeter = cv2.arcLength(target_contour, True)
            approx = cv2.approxPolyDP(target_contour, 0.02 * perimeter, True)

            if len(approx) == 4 and not is_clipped:
                # 完美的四边形！
                is_complete_quad = True
                points = approx.reshape(4, 2)
                cv2.drawContours(frame, [points], -1, (0, 255, 0), 3)  # 绿色框

                # 计算对角线交点作为中心
                frame_center = line_intersection((points[0], points[2]), (points[1], points[3]))
                if frame_center is None:
                    frame_center = (x_rect + w_rect // 2, y_rect + h_rect // 2)

        # 如果不是完整四边形，尝试重建或使用其他方法
        if not is_complete_quad:
            if target_contour is not None:
                # 尝试从部分轮廓重建完整四边形
                reconstructed_points = _reconstruct_quad_from_partial_contour(
                    target_contour, A4_ASPECT_RATIO, frame.shape)

                if reconstructed_points is not None:
                    cv2.drawContours(frame, [reconstructed_points], -1, (255, 0, 0), 2)  # 蓝色框
                    p1, p2, p3, p4 = reconstructed_points
                    frame_center = line_intersection((p1, p3), (p2, p4))
                    if frame_center is None:
                        # 如果对角线交点计算失败，用质心
                        M = cv2.moments(reconstructed_points)
                        if M["m00"] != 0:
                            frame_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                else:
                    # 重建失败，就用原始轮廓
                    cv2.drawContours(frame, [target_contour], -1, (255, 165, 0), 2)  # 橙色框

            # 边界检测算法（备用方案）
            if ENABLE_BORDER_DETECTION and frame_center is None:
                def has_valid_line(strip, side, min_area):
                    strip_contours, _ = cv2.findContours(strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in strip_contours:
                        if cv2.contourArea(c) < min_area: continue
                        x_c, y_c, w_c, h_c = cv2.boundingRect(c)
                        if w_c == 0 or h_c == 0: continue
                        if side in ['left', 'right']:
                            if h_c / w_c > BORDER_LINE_ORIENTATION_RATIO: return True
                        elif side in ['top', 'bottom']:
                            if w_c / h_c > BORDER_LINE_ORIENTATION_RATIO: return True
                    return False

                top_strip = thresh[0:BORDER_CHECK_STRIP_WIDTH, :]
                bottom_strip = thresh[h - BORDER_CHECK_STRIP_WIDTH:h, :]
                left_strip = thresh[:, 0:BORDER_CHECK_STRIP_WIDTH]
                right_strip = thresh[:, w - BORDER_CHECK_STRIP_WIDTH:w]

                has_top_line = has_valid_line(top_strip, 'top', TB_BORDER_CONTOUR_MIN_AREA)
                has_bottom_line = has_valid_line(bottom_strip, 'bottom', TB_BORDER_CONTOUR_MIN_AREA)
                has_left_line = has_valid_line(left_strip, 'left', LR_BORDER_CONTOUR_MIN_AREA)
                has_right_line = has_valid_line(right_strip, 'right', LR_BORDER_CONTOUR_MIN_AREA)

                if has_top_line and has_left_line: frame_center = (FIXED_POINT_OFFSET, FIXED_POINT_OFFSET)
                elif has_top_line and has_right_line: frame_center = (w - FIXED_POINT_OFFSET, FIXED_POINT_OFFSET)
                elif has_bottom_line and has_left_line: frame_center = (FIXED_POINT_OFFSET, h - FIXED_POINT_OFFSET)
                elif has_bottom_line and has_right_line: frame_center = (w - FIXED_POINT_OFFSET, h - FIXED_POINT_OFFSET)
                elif has_left_line: frame_center = (FIXED_POINT_OFFSET, h // 2)
                elif has_right_line: frame_center = (w - FIXED_POINT_OFFSET, h // 2)
                elif has_top_line: frame_center = (w // 2, FIXED_POINT_OFFSET)
                elif has_bottom_line: frame_center = (w // 2, h - FIXED_POINT_OFFSET)

        # 画目标中心点
        if frame_center:
            cv2.circle(frame, frame_center, 7, (0, 0, 255), -1)
            cv2.putText(frame, f"Center: {frame_center}", (frame_center[0] - 60, frame_center[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 模拟激光点
        laser_position = None
        if SIMULATE_LASER_DOT:
            h, w = frame.shape[:2]
            image_center_x, image_center_y = w // 2, h // 2
            laser_pos_x = image_center_x + LASER_OFFSET_X_PIXELS
            laser_pos_y = image_center_y + LASER_OFFSET_Y_PIXELS
            laser_position = (laser_pos_x, laser_pos_y)

            # 画激光点和图像中心的十字线
            cv2.circle(frame, laser_position, 5, LASER_DOT_COLOR_BGR, -1)
            cv2.line(frame, (image_center_x - 10, image_center_y), (image_center_x + 10, image_center_y), (255, 255, 0), 1)
            cv2.line(frame, (image_center_x, image_center_y - 10), (image_center_x, image_center_y + 10), (255, 255, 0), 1)

        return frame_center, laser_position, frame, gray, thresh

    def get_current_data(self):
        """
        获取当前帧的检测结果
        返回：目标中心、激光点位置、处理后的图像
        """
        if not self.cap or not self.cap.isOpened():
            print("错误：摄像头没有初始化")
            return None, None, None

        ret, frame = self.cap.read()
        if not ret:
            print("无法从摄像头读取图像")
            return None, None, None

        target_center, laser_pos, frame_processed, _, _ = self._process_frame(frame)
        return target_center, laser_pos, frame_processed

    def find_center_from_image_file(self, image_path):
        """
        从图片文件中检测目标
        主要用于调试和测试
        """
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"错误: 无法读取图片 '{image_path}'")
            return None, None, None, None

        # 处理图片时暂时关闭自适应曝光
        global ADAPTIVE_EXPOSURE
        original_setting = ADAPTIVE_EXPOSURE
        ADAPTIVE_EXPOSURE = False

        center, laser_pos, result_image, gray_image, thresh_image = self._process_frame(frame)

        # 恢复设置
        ADAPTIVE_EXPOSURE = original_setting
        return center, laser_pos, result_image, gray_image, thresh_image

    def run_debug(self, headless=False):
        """
        调试模式，显示实时检测结果
        """
        print("开始调试... 按 'q' 退出")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取视频帧")
                    break

                center, laser_pos, frame_with_results, gray_frame, thresh_frame = self._process_frame(frame)

                # 显示状态信息
                if center:
                    print(f"\r检测到目标: {center}    ", end="")
                    if ADAPTIVE_EXPOSURE:
                        print(f"| 亮度: {self.mean_brightness:.1f} | 曝光: {self.last_set_exposure}", end="")
                else:
                    print("\r正在搜索目标...                    ", end="")

                # 显示图像（如果不是无头模式）
                if not headless:
                    gray_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                    thresh_bgr = cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2BGR)
                    combined_view = np.hstack((frame_with_results, gray_bgr, thresh_bgr))
                    display_frame = resize_for_display(combined_view, max_width=1920, max_height=600)
                    cv2.imshow("Debug View", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            self.release()
            if not headless:
                cv2.destroyAllWindows()
            print("\n调试结束")

    def release(self):
        """释放摄像头资源"""
        if self.cap:
            self.cap.release()

# 如果直接运行这个文件，就进入测试模式
if __name__ == '__main__':
    if RUN_MODE == 'image':
        print(f"图片测试模式，处理图片: '{IMAGE_PATH_FOR_DEBUG}'")
        detector = FrameDetector(use_camera=False)
        center, laser_pos, result_image, gray_image, thresh_image = detector.find_center_from_image_file(IMAGE_PATH_FOR_DEBUG)

        if result_image is not None:
            if center:
                print(f"找到目标位置: {center}")
            else:
                print("没有检测到目标")

            # 显示结果
            gray_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            thresh_bgr = cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2BGR)
            combined_view = np.hstack((result_image, gray_bgr, thresh_bgr))
            display_image = resize_for_display(combined_view, max_width=1920, max_height=600)
            cv2.imshow("Debug View", display_image)
            print("按任意键关闭窗口。")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif RUN_MODE == 'camera':
        mode_str = "无头模式" if HEADLESS_MODE else "带显示窗口"
        print(f"摄像头模式 ({mode_str})")
        try:
            detector = FrameDetector()
            detector.run_debug(headless=HEADLESS_MODE)
        except IOError as e:
            print(e)
        except KeyboardInterrupt:
            print("\n程序被中断")
    else:
        print(f"错误: 未知的运行模式 '{RUN_MODE}'，请设置为 'camera' 或 'image'")
