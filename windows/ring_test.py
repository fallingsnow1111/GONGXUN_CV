import cv2 as cv
import numpy as np
import sys
import time

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_INDEX = 0

COLOR_HSV = {
    "red": {
        "lower1": np.array([0, 100, 100], np.uint8),
        "upper1": np.array([10, 255, 255], np.uint8),
        "lower2": np.array([160, 100, 100], np.uint8),
        "upper2": np.array([180, 255, 255], np.uint8)
    },
    "green": {
        "lower": np.array([50, 30, 128], np.uint8),
        "upper": np.array([87, 255, 255], np.uint8)
    },
    "blue": {
        "lower": np.array([92, 71, 88], np.uint8),
        "upper": np.array([122, 255, 255], np.uint8)
    }
}

RING_RULES = {
    "area_min": 2000,
    "area_max": 90000,
    "circularity_min": 0.45,
    "fill_ratio_max": 0.82,
    "erode_iter": 1,
    "dilate_iter": 2
}

kernel = np.ones((3, 3), np.uint8)

MODE_MAP = {
    0: [],
    1: ["red"],
    2: ["green"],
    3: ["blue"],
    4: ["red", "green", "blue"],
}

def open_camera():
    if sys.platform.startswith("win"):
        # DirectShow是Windows平台上OpenCV提供的一种视频捕获接口，使用它可以更快地访问摄像头并减少延迟。
        cap = cv.VideoCapture(CAMERA_INDEX, cv.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv.VideoCapture(CAMERA_INDEX, cv.CAP_ANY)
    else:
        cap = cv.VideoCapture(CAMERA_INDEX, cv.CAP_V4L2)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return cap
    
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv.CAP_PROP_FPS, 30)

    time.sleep(0.5)

    ret, frame = cap.read()

    if ret and frame is not None:
        print("Camera opened successfully.FPS:", cap.get(cv.CAP_PROP_FPS))
    else:
        print("Error: Could not read from camera.")
    
    # 返回摄像头对象，供后续使用
    return cap

def get_color_mask(hsv_img, color):
    cfg = COLOR_HSV[color]
    if color == "red":
        # 二值化处理，得到红色区域的掩码
        mask1 = cv.inRange(hsv_img, cfg["lower1"], cfg["upper1"])
        mask2 = cv.inRange(hsv_img, cfg["lower2"], cfg["upper2"])
        mask = cv.bitwise_or(mask1, mask2)
    else:
        mask = cv.inRange(hsv_img, cfg["lower"], cfg["upper"])

    mask_eroded = cv.erode(mask, kernel, iterations=RING_RULES["erode_iter"])
    mask_final = cv.dilate(mask_eroded, kernel, iterations=RING_RULES["dilate_iter"])

    return mask_final

def detect_rings(mask, color):
    # 查找轮廓，返回值是一个包含所有轮廓的列表和一个层级信息的数组
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    best_ring = None
    best_score = -1

    for i , cnt in enumerate(contours):
        # 计算轮廓的面积，单位是像素
        area = cv.contourArea(cnt)
        if not (RING_RULES["area_min"] < area < RING_RULES["area_max"]):
            continue

        peri = cv.arcLength(cnt, True)
        if peri <= 0:
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < RING_RULES["circularity_min"]:
            continue

        # moments返回一个字典，包含圆心坐标、半径、面积、圆度、填充率等信息
        m = cv.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

        (x, y), radius = cv.minEnclosingCircle(cnt)
        circle_area = np.pi * radius * radius
        fill_ratio = area / circle_area

        has_hole = hierarchy is not None and hierarchy[0][i][2] != -1
        if not has_hole and fill_ratio > RING_RULES["fill_ratio_max"]:
            continue
        
        current_score = area * circularity
        if current_score > best_score:
            best_score = current_score
            best_ring = {
                "color": color,
                "contour": cnt,
                "center": (cx, cy),
                "radius": int(radius),
                "area": area,
            }

    return best_ring

def draw_ring(frame, ring):
    if not ring:
        return frame
    
    color = ring["color"]
    cx, cy = ring["center"]
    radius = ring["radius"]

    color_draw = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0)
    }

    # 色环轮廓是实际识别出来的色环边缘
    cv.drawContours(frame, [ring["contour"]], -1, color_draw[color], 2)
    # 色环外接圆是计算出来的一个圆形，理论上应该包含整个色环
    cv.circle(frame, (cx, cy), radius, color_draw[color], 1, cv.LINE_AA)
    cv.circle(frame, (cx, cy), 3, (0, 255, 255), -1)  # 圆心标记为黄色小圆点
    cv.putText(
        frame,
        f"{color.capitalize()} Ring",
        (max(10, cx - 60), max(20, cy - 15)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        color_draw[color],
        2
    )

    return frame

def ring_recognition(cap):
    mode = 4
    print("Mode keys: 0=off, 1=red ring, 2=green ring, 3=blue ring, 4=all rings, q=quit")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Could not read frame from camera.")
            break

        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        active_colors = MODE_MAP.get(mode, [])
        combined_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)

        for color in active_colors:
            color_mask = get_color_mask(hsv_frame, color)
            combined_mask = cv.bitwise_or(combined_mask, color_mask)
            ring = detect_rings(color_mask, color)
            frame = draw_ring(frame, ring)

        if mode == 0:
            cv.putText(
                frame,
                "Mode 0: OFF",
                (20, 35),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        else:
            mode_text = {
                1: "Mode 1: RED RING",
                2: "Mode 2: GREEN RING",
                3: "Mode 3: BLUE RING",
                4: "Mode 4: ALL RINGS",
            }.get(mode, f"Mode {mode}")
            cv.putText(
                frame,
                mode_text,
                (20, 35),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        cv.imshow("Ring Mask", combined_mask)

        cv.imshow("Ring Detection", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break
        if ord('0') <= key <= ord('4'):
            mode = key - ord('0')
            print(f"Mode changed to {mode}")

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    cap = open_camera()
    if cap.isOpened():
        ring_recognition(cap)
    else:
        print("Failed to open camera. Exiting.")