# 阈值实时调整工具，支持摄像头或静态图像输入，实时预览掩膜效果，并可保存当前配置和结果。
import argparse
import datetime
import json
from pathlib import Path

import cv2 as cv
import numpy as np


WINDOW_TUNER = "Tuner"
WINDOW_MASK = "Mask_Effect"
WINDOW_PREVIEW = "Preview"


def nothing(_):
    pass


def build_parser():
    parser = argparse.ArgumentParser(
        description="Real-time HSV threshold tuning with trackbars."
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=1920, help="Capture width")
    parser.add_argument("--height", type=int, default=1080, help="Capture height")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS")
    parser.add_argument("--image", type=str, default="", help="Optional static image path")
    parser.add_argument("--h-low", type=int, default=50)
    parser.add_argument("--s-low", type=int, default=30)
    parser.add_argument("--v-low", type=int, default=80)
    parser.add_argument("--h-high", type=int, default=90)
    parser.add_argument("--s-high", type=int, default=255)
    parser.add_argument("--v-high", type=int, default=255)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="gongxun/windows/debug_samples",
        help="Directory for saving tuning outputs",
    )
    return parser


def open_camera(index, width, height, fps):
    cap = cv.VideoCapture(index, cv.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv.VideoCapture(index, cv.CAP_ANY)
    if not cap.isOpened():
        return cap

    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FPS, fps)
    return cap


def create_trackbars(args):
    cv.namedWindow(WINDOW_TUNER)
    cv.createTrackbar("H_low", WINDOW_TUNER, int(np.clip(args.h_low, 0, 179)), 179, nothing)
    cv.createTrackbar("H_high", WINDOW_TUNER, int(np.clip(args.h_high, 0, 179)), 179, nothing)
    cv.createTrackbar("S_low", WINDOW_TUNER, int(np.clip(args.s_low, 0, 255)), 255, nothing)
    cv.createTrackbar("S_high", WINDOW_TUNER, int(np.clip(args.s_high, 0, 255)), 255, nothing)
    cv.createTrackbar("V_low", WINDOW_TUNER, int(np.clip(args.v_low, 0, 255)), 255, nothing)
    cv.createTrackbar("V_high", WINDOW_TUNER, int(np.clip(args.v_high, 0, 255)), 255, nothing)

    cv.createTrackbar("Blur(odd)", WINDOW_TUNER, 1, 20, nothing)
    cv.createTrackbar("Erode", WINDOW_TUNER, 0, 8, nothing)
    cv.createTrackbar("Dilate", WINDOW_TUNER, 0, 8, nothing)


def read_trackbar_values():
    values = {
        "h_low": cv.getTrackbarPos("H_low", WINDOW_TUNER),
        "h_high": cv.getTrackbarPos("H_high", WINDOW_TUNER),
        "s_low": cv.getTrackbarPos("S_low", WINDOW_TUNER),
        "s_high": cv.getTrackbarPos("S_high", WINDOW_TUNER),
        "v_low": cv.getTrackbarPos("V_low", WINDOW_TUNER),
        "v_high": cv.getTrackbarPos("V_high", WINDOW_TUNER),
        "blur": cv.getTrackbarPos("Blur(odd)", WINDOW_TUNER),
        "erode": cv.getTrackbarPos("Erode", WINDOW_TUNER),
        "dilate": cv.getTrackbarPos("Dilate", WINDOW_TUNER),
    }

    # Prevent invalid lower > upper ranges.
    if values["h_low"] > values["h_high"]:
        values["h_low"], values["h_high"] = values["h_high"], values["h_low"]
    if values["s_low"] > values["s_high"]:
        values["s_low"], values["s_high"] = values["s_high"], values["s_low"]
    if values["v_low"] > values["v_high"]:
        values["v_low"], values["v_high"] = values["v_high"], values["v_low"]

    if values["blur"] % 2 == 0:
        values["blur"] += 1
    values["blur"] = max(1, values["blur"])
    return values


def apply_threshold(frame, values):
    work = frame.copy()
    if values["blur"] > 1:
        work = cv.GaussianBlur(work, (values["blur"], values["blur"]), 0)

    hsv = cv.cvtColor(work, cv.COLOR_BGR2HSV)
    lower = np.array([values["h_low"], values["s_low"], values["v_low"]], np.uint8)
    upper = np.array([values["h_high"], values["s_high"], values["v_high"]], np.uint8)

    mask = cv.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    if values["erode"] > 0:
        mask = cv.erode(mask, kernel, iterations=values["erode"])
    if values["dilate"] > 0:
        mask = cv.dilate(mask, kernel, iterations=values["dilate"])

    overlay = frame.copy()
    overlay[mask > 0] = (0, 255, 0)
    preview = cv.addWeighted(frame, 0.65, overlay, 0.35, 0)
    return mask, preview, lower, upper


def save_outputs(save_dir, frame, mask, values):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    frame_path = save_dir / f"tuner_frame_{ts}.jpg"
    mask_path = save_dir / f"tuner_mask_{ts}.png"
    json_path = save_dir / f"tuner_threshold_{ts}.json"

    cv.imwrite(str(frame_path), frame)
    cv.imwrite(str(mask_path), mask)

    payload = {
        "lower": [values["h_low"], values["s_low"], values["v_low"]],
        "upper": [values["h_high"], values["s_high"], values["v_high"]],
        "blur": values["blur"],
        "erode": values["erode"],
        "dilate": values["dilate"],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved frame: {frame_path}")
    print(f"Saved mask: {mask_path}")
    print(f"Saved threshold json: {json_path}")


def main():
    args = build_parser().parse_args()
    create_trackbars(args)

    static_image = None
    if args.image:
        static_image = cv.imread(args.image)
        if static_image is None:
            print(f"Failed to load image: {args.image}")
            return
        print(f"Tuning with static image: {args.image}")

    cap = None
    if static_image is None:
        cap = open_camera(args.camera_index, args.width, args.height, args.fps)
        if not cap.isOpened():
            print("Camera open failed")
            return
        print(f"Tuning with live camera index={args.camera_index}")

    save_dir = Path(args.save_dir)

    print("\nControls:")
    print("  p : print current threshold config")
    print("  s : save current frame/mask/json")
    print("  q : quit")

    while True:
        if static_image is None:
            assert cap is not None
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera frame read failed")
                break
        else:
            frame = static_image.copy()

        values = read_trackbar_values()
        mask, preview, lower, upper = apply_threshold(frame, values)

        text = (
            f"L=({lower[0]},{lower[1]},{lower[2]}) "
            f"U=({upper[0]},{upper[1]},{upper[2]}) "
            f"B={values['blur']} E={values['erode']} D={values['dilate']}"
        )
        cv.putText(preview, text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv.imshow(WINDOW_MASK, mask)
        cv.imshow(WINDOW_PREVIEW, preview)

        key = cv.waitKey(1 if static_image is None else 20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            print("\nCurrent threshold:")
            print(f"  lower=np.array([{lower[0]}, {lower[1]}, {lower[2]}], np.uint8)")
            print(f"  upper=np.array([{upper[0]}, {upper[1]}, {upper[2]}], np.uint8)")
            print(f"  blur={values['blur']} erode={values['erode']} dilate={values['dilate']}")
        if key == ord("s"):
            save_outputs(save_dir, frame, mask, values)

    if cap is not None:
        cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
