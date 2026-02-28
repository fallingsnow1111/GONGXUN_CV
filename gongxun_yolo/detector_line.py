"""
直线检测模块 - action 9（传统 CV，保留 HSV 方案）
白色直线 + 绿色圆环 同时检测
卡尔曼滤波平滑白线斜率，减少抖动和短暂丢线
"""
import numpy as np
import cv2 as cv

from config import (
    HSV_THRESHOLDS, KERNEL_3x3,
    AREA_MIN, AREA_MAX, CX_SCALE, CY_SCALE,
)


# ==================== 卡尔曼滤波器 ====================
class LineKalmanFilter:
    """
    白线斜率卡尔曼滤波器
    状态: [slope, d_slope]  (斜率 + 斜率变化率)
    观测: [slope]
    """

    def __init__(self):
        self.kf = cv.KalmanFilter(2, 1, 0)

        # 状态转移: slope_k = slope_{k-1} + d_slope_{k-1}
        self.kf.transitionMatrix = np.array([
            [1, 1],
            [0, 1],
        ], dtype=np.float32)

        # 观测矩阵: 只观测 slope
        self.kf.measurementMatrix = np.array([[1, 0]], dtype=np.float32)

        # 过程噪声 — 越小越平滑，弯道跟踪会滞后
        self.kf.processNoiseCov = np.array([
            [1e-3, 0],
            [0, 1e-4],
        ], dtype=np.float32)

        # 观测噪声 — 越大越不信单帧测量，输出越平滑
        self.kf.measurementNoiseCov = np.array([[5e-2]], dtype=np.float32)

        self.kf.errorCovPost = np.eye(2, dtype=np.float32)
        self.kf.statePost = np.zeros((2, 1), dtype=np.float32)

        self._initialized = False
        self._miss_count = 0
        self._max_miss = 5  # 连续丢失超过此帧数停止输出

    def update(self, slope=None):
        """
        每帧调用。slope=None 表示该帧未检测到线。
        返回滤波后的 slope，或 None（丢失太久）。
        """
        prediction = self.kf.predict()

        if slope is not None:
            if not self._initialized:
                self.kf.statePost = np.array(
                    [[slope], [0]], dtype=np.float32
                )
                self._initialized = True
                return slope

            measurement = np.array([[np.float32(slope)]])
            corrected = self.kf.correct(measurement)
            self._miss_count = 0
            return float(corrected[0, 0])
        else:
            self._miss_count += 1
            if self._miss_count > self._max_miss or not self._initialized:
                return None
            return float(prediction[0, 0])


# 模块级单例，跨帧保持状态
_line_kf = LineKalmanFilter()


# ==================== 检测函数 ====================
def _detect_white_line(hsv, frame, action_flag, serial_comm):
    """检测白色直线，卡尔曼滤波平滑后发送斜率"""
    th = HSV_THRESHOLDS['white']
    white_mask = cv.inRange(hsv, th['lower'], th['upper'])
    edges = cv.Canny(white_mask, 50, 150)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 30,
                           minLineLength=40, maxLineGap=10)

    raw_slope = None
    x1 = y1 = x2 = y2 = 0

    if lines is not None and lines.size > 0:
        longest = max(lines, key=lambda l: np.linalg.norm(l[0][2:] - l[0][:2]))
        x1, y1, x2, y2 = longest[0]
        dx, dy = x2 - x1, y2 - y1
        raw_slope = (dy / dx) if dx != 0 else 10.0

    # 卡尔曼滤波
    filtered_slope = _line_kf.update(raw_slope)

    if filtered_slope is not None:
        # 有原始检测时画线
        if raw_slope is not None:
            cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # OSD: 同时显示原始值和滤波值，方便调参
        raw_str = f"{raw_slope:.2f}" if raw_slope is not None else "N/A"
        cv.putText(frame, f"raw={raw_str}  kf={filtered_slope:.2f}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        serial_comm.send_line(action_flag, int(filtered_slope * 1000))


def _detect_green_circle(hsv, frame, action_flag, serial_comm):
    """检测绿色圆环"""
    th = HSV_THRESHOLDS['green']
    green_mask = cv.inRange(hsv, th['lower'], th['upper'])
    eroded = cv.erode(green_mask, KERNEL_3x3, iterations=1)
    dilated = cv.dilate(eroded, KERNEL_3x3, iterations=4)

    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return

    largest = max(contours, key=cv.contourArea)
    area = cv.contourArea(largest)
    if not (AREA_MIN < area < AREA_MAX):
        return

    (x, y), radius = cv.minEnclosingCircle(largest)
    center = (int(x), int(y))
    radius = int(radius)

    cv.circle(frame, center, radius, (255, 0, 0), 2)
    cv.circle(frame, center, 2, (0, 255, 0), 3)
    cv.putText(frame, "Green Circle", (center[0] - 50, center[1] - radius - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    serial_comm.send_circle(action_flag, int(center[0] / CX_SCALE), int(center[1] / CY_SCALE))


def detect(frame, action_flag, serial_comm):
    """
    直线 + 绿色圆环检测（action=9）。

    Args:
        frame:       BGR 图像
        action_flag: 当前模式 (9)
        serial_comm: SerialComm 实例

    Returns:
        标注后的 frame
    """
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    _detect_white_line(hsv, frame, action_flag, serial_comm)
    _detect_green_circle(hsv, frame, action_flag, serial_comm)
    return frame
