"""
HSV 传统检测模块 - YOLO 模型不可用时的回退方案 (action 1~8)
"""
import numpy as np
import cv2 as cv

from config import (
    HSV_THRESHOLDS, ACTION_TO_HSV_COLOR, ACTION_DRAW_COLOR,
    AREA_MIN, AREA_MAX, KERNEL_3x3, CX_SCALE, CY_SCALE,
)


def _get_color_mask(hsv_img, color_key):
    """根据颜色键名生成 HSV 掩码"""
    th = HSV_THRESHOLDS[color_key]
    if color_key == 'red':
        return (cv.inRange(hsv_img, th['lower1'], th['upper1'])
                + cv.inRange(hsv_img, th['lower2'], th['upper2']))
    return cv.inRange(hsv_img, th['lower'], th['upper'])


def _morphology(mask, action_flag):
    """形态学处理"""
    eroded = cv.erode(mask, KERNEL_3x3, iterations=1)
    iters = 1 if action_flag == 5 else 4
    return cv.dilate(eroded, KERNEL_3x3, iterations=iters)


def detect(frame, action_flag, serial_comm):
    """
    传统 HSV 颜色检测（回退方案）。

    Args:
        frame:       BGR 图像
        action_flag: 当前模式 (1-8)
        serial_comm: SerialComm 实例

    Returns:
        标注后的 frame
    """
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    color_key = ACTION_TO_HSV_COLOR.get(action_flag)
    draw_color = ACTION_DRAW_COLOR.get(action_flag, (255, 255, 255))

    if color_key is None:
        serial_comm.send_empty(action_flag)
        return frame

    mask = _get_color_mask(hsv, color_key)
    filtered = _morphology(mask, action_flag)

    contours, _ = cv.findContours(filtered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    found = False

    for cnt in sorted(contours, key=cv.contourArea, reverse=True)[:5]:
        area = cv.contourArea(cnt)
        if not (AREA_MIN < area < AREA_MAX):
            continue
        perimeter = cv.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.5:
            continue

        M = cv.moments(cnt)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cv.drawContours(frame, [cnt], -1, draw_color, 2)
        cv.circle(frame, (cX, cY), 5, (0, 0, 0), -1)

        cx_send = int(cX / CX_SCALE)
        cy_send = int(cY / CY_SCALE)
        serial_comm.send_circle(action_flag, cx_send, cy_send)
        print(f"[HSV] flag={action_flag} x={cx_send} y={cy_send}")
        found = True
        break  # 只取最大

    if not found:
        serial_comm.send_empty(action_flag)

    return frame
