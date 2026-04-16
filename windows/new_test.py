import argparse
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np


CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_WARMUP_SECONDS = 0.5


def open_camera(camera_index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
        if not cap.isOpened():
            return cap
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    else:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            return cap

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    time.sleep(CAMERA_WARMUP_SECONDS)

    ok, frame = cap.read()
    if ok and frame is not None and frame.size > 0:
        print(f"Camera opened: index={camera_index}, shape={frame.shape}")
    else:
        print("Camera opened but first frame invalid")

    return cap


def color_circle_position(
    img: np.ndarray,
    show_debug: bool = True,
) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    Ring detection helper.
    Args:
        img: Input BGR image.
    Returns:
        (x1, y1, x2, y2, x3, y3) sorted by x, or None if not enough circles.
    """
    erode_kernel = np.ones((3, 3), np.uint8)
    erode_hsv = cv2.erode(img, erode_kernel, iterations=2)
    kernel = np.ones((7, 7), np.uint8)
    dirange_hsv = cv2.dilate(erode_hsv, kernel, iterations=1)
    gray_img = cv2.cvtColor(dirange_hsv, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    clahed = clahe.apply(gray_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gradient = cv2.morphologyEx(clahed, cv2.MORPH_GRADIENT, kernel)

    result = cv2.GaussianBlur(gradient, (7, 7), sigmaX=3, sigmaY=3)
    eqal_img = cv2.convertScaleAbs(result, alpha=4, beta=0)
    eqal_img = cv2.GaussianBlur(eqal_img, (7, 7), sigmaX=3, sigmaY=3)

    _, threshold_img = cv2.threshold(eqal_img, 70, 255, cv2.THRESH_BINARY)
    threshold_img = cv2.GaussianBlur(threshold_img, (9, 9), sigmaX=3, sigmaY=3)

    circles = cv2.HoughCircles(
        threshold_img,
        cv2.HOUGH_GRADIENT_ALT,
        1.5,
        50,
        param1=100,
        param2=0.95,
        minRadius=15,
        maxRadius=50,
    )

    if show_debug:
        cv2.imshow("video", gray_img)
        cv2.imshow("video2", eqal_img)
        cv2.imshow("video3", threshold_img)

    if circles is None:
        return None

    circles_arr = np.squeeze(circles, axis=0)
    if circles_arr.ndim != 2 or circles_arr.shape[0] < 3 or circles_arr.shape[1] < 3:
        return None

    # Keep the three strongest circles by radius, then sort by x position.
    circles_rounded = np.rint(circles_arr[:, :3]).astype(np.int32)
    sorted_by_radius = sorted(circles_rounded.tolist(), key=lambda c: c[2], reverse=True)
    circle_list = sorted(sorted_by_radius[:3], key=lambda c: c[0])

    for circle in circle_list:
        cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
        cv2.circle(img, (circle[0], circle[1]), 2, (255, 0, 0), 2)

    return (
        int(circle_list[0][0]),
        int(circle_list[0][1]),
        int(circle_list[1][0]),
        int(circle_list[1][1]),
        int(circle_list[2][0]),
        int(circle_list[2][1]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-ring center detection with HoughCircles")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument("--fps", type=int, default=CAMERA_FPS)
    parser.add_argument("--no-debug", action="store_true", help="Disable preprocessing windows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cap = open_camera(args.camera_index, args.width, args.height, args.fps)
    if not cap.isOpened():
        print("Camera open failed. Try: python windows/find_camera_id.py")
        return

    print("Press q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break
        if frame is None or frame.size == 0:
            continue

        show = frame.copy()
        centers = color_circle_position(show, show_debug=not args.no_debug)

        if centers is not None:
            x1, y1, x2, y2, x3, y3 = centers
            text = f"Centers: ({x1},{y1}) ({x2},{y2}) ({x3},{y3})"
            cv2.putText(show, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(text)
        else:
            cv2.putText(show, "Not enough circles (need 3)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("new_test_detection", show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
