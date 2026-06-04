"""
白线检测模块（mode 9）— 保留传统 Hough 变换，不用 YOLO
从原方案 color_line_det.py 的 find_lines() 提取
"""
import numpy as np
import cv2 as cv

# 白色 HSV 阈值
WHITE_LOWER = np.array([0,   0,   200], np.uint8)
WHITE_UPPER = np.array([180, 255, 255], np.uint8)


class WhiteLineDetector:
    def detect(self, frame):
        """
        检测画面中的白线。
        返回: ((x1, y1, x2, y2), slope) 或 None（未检测到）
        slope 为斜率，×1000 后传入 send_line_data
        """
        hsv   = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask  = cv.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
        edges = cv.Canny(mask, 50, 150)
        lines = cv.HoughLinesP(edges, 1, np.pi / 180, 30,
                               minLineLength=40, maxLineGap=10)

        if lines is None or len(lines) == 0:
            return None

        # 取最长线段
        def line_len(l):
            x1, y1, x2, y2 = l[0]
            return (x2 - x1) ** 2 + (y2 - y1) ** 2

        longest = max(lines, key=line_len)
        x1, y1, x2, y2 = longest[0]
        dx = x2 - x1
        slope = (y2 - y1) / dx if dx != 0 else float('inf')
        return (x1, y1, x2, y2), slope
