# HSV范围采样工具，支持摄像头实时预览和静态图像采样，点击像素获取HSV值并推荐阈值区间。
import argparse
import datetime
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np


WINDOW_LIVE = "Live"
WINDOW_SAMPLE = "Sample"
CAMERA_WARMUP_SECONDS = 0.5


def build_parser():
    parser = argparse.ArgumentParser(
        description="Capture static samples and click pixels to estimate HSV ranges."
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=1920, help="Capture width")
    parser.add_argument("--height", type=int, default=1080, help="Capture height")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="windows/debug_samples",
        help="Directory used to store captured frames",
    )
    return parser


def _open_single_camera(index, width, height, fps):
    backend = "CAP_V4L2"
    if sys.platform.startswith("win"):
        cap = cv.VideoCapture(index, cv.CAP_DSHOW)
        backend = "CAP_DSHOW"
        if not cap.isOpened():
            cap = cv.VideoCapture(index, cv.CAP_ANY)
            backend = "CAP_ANY"
        if not cap.isOpened():
            return None
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc(*"MJPG"))
    else:
        cap = cv.VideoCapture(index, cv.CAP_V4L2)
        if not cap.isOpened():
            return None

    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FPS, fps)
    time.sleep(CAMERA_WARMUP_SECONDS)

    ok, frame = False, None
    for _ in range(10):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            break
        time.sleep(0.05)

    if not ok or frame is None or frame.size == 0:
        cap.release()
        return None

    real_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv.CAP_PROP_FPS)
    print(
        f"Camera opened: index={index}, backend={backend}, "
        f"size={real_w}x{real_h}, fps={real_fps:.1f}, frame_shape={frame.shape}"
    )
    return cap


def open_camera(index, width, height, fps):
    candidates = [index] + [i for i in range(6) if i != index]
    for idx in candidates:
        cap = _open_single_camera(idx, width, height, fps)
        if cap is not None and cap.isOpened():
            if idx != index:
                print(f"Camera index {index} failed, fallback to index {idx}")
            return cap, idx
    return cv.VideoCapture(), index


def clamp_range(low, high, channel):
    if channel == "h":
        lo = int(max(0, min(179, low)))
        hi = int(max(0, min(179, high)))
    else:
        lo = int(max(0, min(255, low)))
        hi = int(max(0, min(255, high)))
    return lo, hi


def print_recommended_threshold(points_hsv):
    hsv_arr = np.array(points_hsv, dtype=np.int32)
    h_min, s_min, v_min = hsv_arr.min(axis=0)
    h_max, s_max, v_max = hsv_arr.max(axis=0)

    # Give a practical margin so online tuning starts from a usable interval.
    h_lo, h_hi = clamp_range(h_min - 5, h_max + 5, "h")
    s_lo, s_hi = clamp_range(s_min - 30, s_max + 30, "sv")
    v_lo, v_hi = clamp_range(v_min - 30, v_max + 30, "sv")

    print("\n[HSV Samples Summary]")
    print(f"  sample_count={len(points_hsv)}")
    print(f"  raw_min=({h_min}, {s_min}, {v_min})")
    print(f"  raw_max=({h_max}, {s_max}, {v_max})")
    print(f"  suggested_lower=np.array([{h_lo}, {s_lo}, {v_lo}], np.uint8)")
    print(f"  suggested_upper=np.array([{h_hi}, {s_hi}, {v_hi}], np.uint8)")


def main():
    args = build_parser().parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cap, used_index = open_camera(args.camera_index, args.width, args.height, args.fps)
    if not cap.isOpened():
        print("Camera open failed. Try: python windows/find_camera_id.py")
        return
    print(f"Using camera index: {used_index}")

    state = {
        "live_bgr": None,
        "live_hsv": None,
        "sample_bgr": None,
        "sample_hsv": None,
        "clicked_points": [],
    }

    def on_mouse_click(event, x, y, flags, param):
        if event != cv.EVENT_LBUTTONDOWN:
            return

        sample_hsv = state["sample_hsv"]
        sample_bgr = state["sample_bgr"]
        if sample_hsv is not None and sample_bgr is not None:
            if 0 <= y < sample_hsv.shape[0] and 0 <= x < sample_hsv.shape[1]:
                hsv_pixel = sample_hsv[y, x]
                bgr_pixel = sample_bgr[y, x]
                state["clicked_points"].append(tuple(int(v) for v in hsv_pixel))
                print(
                    f"Click@Sample ({x}, {y}) BGR={tuple(int(v) for v in bgr_pixel)} "
                    f"HSV={tuple(int(v) for v in hsv_pixel)}"
                )
                print_recommended_threshold(state["clicked_points"])
                return

        live_hsv = state["live_hsv"]
        if live_hsv is not None and 0 <= y < live_hsv.shape[0] and 0 <= x < live_hsv.shape[1]:
            hsv_pixel = live_hsv[y, x]
            print(f"Click@Live ({x}, {y}) HSV={tuple(int(v) for v in hsv_pixel)}")

    cv.namedWindow(WINDOW_LIVE)
    cv.namedWindow(WINDOW_SAMPLE)
    cv.setMouseCallback(WINDOW_LIVE, on_mouse_click)
    cv.setMouseCallback(WINDOW_SAMPLE, on_mouse_click)

    print("\nControls:")
    print("  s : save current frame as static sample")
    print("  c : clear clicked HSV points")
    print("  q : quit")
    print("  Mouse left click on Live/Sample window to read HSV")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Camera frame read failed")
            break

        state["live_bgr"] = frame
        state["live_hsv"] = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        live_display = frame.copy()
        cv.putText(
            live_display,
            "Press s to capture sample, click to read HSV",
            (20, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv.imshow(WINDOW_LIVE, live_display)

        if state["sample_bgr"] is not None:
            sample_display = state["sample_bgr"].copy()
            cv.putText(
                sample_display,
                f"Sample points: {len(state['clicked_points'])}",
                (20, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv.imshow(WINDOW_SAMPLE, sample_display)
        else:
            blank = np.zeros((300, 800, 3), dtype=np.uint8)
            cv.putText(
                blank,
                "No sample yet. Press s to capture a static frame.",
                (20, 150),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv.imshow(WINDOW_SAMPLE, blank)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            state["clicked_points"].clear()
            print("Cleared sampled HSV points")
        if key == ord("s"):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = save_dir / f"sample_{ts}.jpg"
            cv.imwrite(str(img_path), frame)
            state["sample_bgr"] = frame.copy()
            state["sample_hsv"] = cv.cvtColor(state["sample_bgr"], cv.COLOR_BGR2HSV)
            state["clicked_points"].clear()
            print(f"Saved sample image: {img_path}")
            print("Now click target and background points in Sample window")

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
