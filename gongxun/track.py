import cv2 as cv
import numpy as np
import time
import serial
import threading

idth = 320
height = 240
FPS = 30

cap_1 = cv.VideoCapture(0,cv.CAP_V4L2)  # 车前方的摄像头(未启用)

# 假设这些是你的初始HSV值
HSV_MIN_BLUE_YUAN = (0, 82, 130)
HSV_MAX_BLUE_YUAN = (62, 255,255)

# 创建窗口
cv.namedWindow('Trackbars')

# 创建trackbar回调函数
def on_trackbar(val):
    pass

# 创建trackbar
cv.createTrackbar('Hue Min', 'Trackbars', HSV_MIN_BLUE_YUAN[0], 180, on_trackbar)
cv.createTrackbar('Sat Min', 'Trackbars', HSV_MIN_BLUE_YUAN[1], 255, on_trackbar)
cv.createTrackbar('Val Min', 'Trackbars', HSV_MIN_BLUE_YUAN[2], 255, on_trackbar)
cv.createTrackbar('Hue Max', 'Trackbars', HSV_MAX_BLUE_YUAN[0], 180, on_trackbar)
cv.createTrackbar('Sat Max', 'Trackbars', HSV_MAX_BLUE_YUAN[1], 255, on_trackbar)
cv.createTrackbar('Val Max', 'Trackbars', HSV_MAX_BLUE_YUAN[2], 255, on_trackbar)

# 定义kernel
kernel = np.ones((3, 3), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)

# 你的检测颜色函数
def detect_color_Scan():
    while cap_1.isOpened():
        ret_color, frame_color = cap_1.read()
        if not ret_color:
            break

        # 读取trackbar的值
        h_min = cv.getTrackbarPos('Hue Min', 'Trackbars')
        s_min = cv.getTrackbarPos('Sat Min', 'Trackbars')
        v_min = cv.getTrackbarPos('Val Min', 'Trackbars')
        h_max = cv.getTrackbarPos('Hue Max', 'Trackbars')
        s_max = cv.getTrackbarPos('Sat Max', 'Trackbars')
        v_max = cv.getTrackbarPos('Val Max', 'Trackbars')

        # 使用trackbar的值来创建HSV范围
        lower_color_B_YUAN = np.array([h_min, s_min, v_min], np.uint8)
        upper_color_B_YUAN = np.array([h_max, s_max, v_max], np.uint8)

        # 你的HSV转换和颜色检测代码
        hsv_image = cv.cvtColor(frame_color, cv.COLOR_BGR2HSV)
        median = cv.medianBlur(hsv_image, 3)
        color_mask = cv.inRange(median, lower_color_B_YUAN, upper_color_B_YUAN)
        filtered_mask = cv.dilate(color_mask, kernel2, iterations=4)
        filtered_mask = cv.erode(filtered_mask, kernel2, iterations=2)#2
        filtered_mask = cv.morphologyEx(filtered_mask, cv.MORPH_OPEN, kernel, iterations=1)#1
        filtered_mask = cv.erode(filtered_mask, kernel2, iterations=6)#6
        filtered_mask = cv.dilate(filtered_mask, kernel2, iterations=7)#7

        # 显示结果
        cv.imshow('Color Detection', color_mask)
        cv.imshow('Color Detection2', frame_color)
        cv.waitKey(1)
        time.sleep(0.01)



detect_color_Scan()
