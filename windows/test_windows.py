import importlib
import sys
import threading
import time

import cv2 as cv
import numpy as np

try:
    serial = importlib.import_module("serial")
except ImportError:
    serial = None

# Camera and output config
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
FPS = 30
CAMERA_INDEX = 0
CAMERA_WARMUP_SECONDS = 0.5

OUTPUT_MODE = "print"  # "print" or "serial"
SERIAL_PORT_WINDOWS = "COM5"
SERIAL_PORT_LINUX = "/dev/ttyAMA0"
SERIAL_TIMEOUT = 0.001

SCALE_X = 3.5
SCALE_Y = 2.5

TARGET_RULES = {
    "block": {
        "area_min": 5000,
        "area_max": 50000,
        "circularity_min": 0.5,
        "erode_iter": 1,
        "dilate_iter": 4,
    },
    "ring": {
        "area_min": 2000,
        "area_max": 90000,
        "circularity_min": 0.45,
        "fill_ratio_max": 0.82,
        "erode_iter": 1,
        "dilate_iter": 2,
    },
}

MODE_CONFIG = {
    1: {"target": "block", "color": "red", "draw": (0, 0, 255), "name": "Red Block"},
    2: {"target": "block", "color": "green", "draw": (0, 255, 0), "name": "Green Block"},
    3: {"target": "block", "color": "blue", "draw": (255, 0, 0), "name": "Blue Block"},
    4: {"target": "ring", "color": "red", "draw": (0, 0, 255), "name": "Red Ring"},
    5: {"target": "ring", "color": "green_circle", "draw": (0, 255, 0), "name": "Green Ring"},
    6: {"target": "ring", "color": "blue_circle", "draw": (255, 0, 0), "name": "Blue Ring"},
    7: {"target": "block", "color": "up_green", "draw": (0, 255, 0), "name": "Up Green"},
    8: {"target": "block", "color": "all_color", "draw": (255, 255, 255), "name": "All Color"},
}

# HSV thresholds
COLOR_THRESHOLDS = {
    "red": {
        "lower1": np.array([0, 100, 100], np.uint8),
        "upper1": np.array([10, 255, 255], np.uint8),
        "lower2": np.array([160, 100, 100], np.uint8),
        "upper2": np.array([180, 255, 255], np.uint8),
    },
    # 207光照下的绿色阈值
    "green": {
        "lower": np.array([19, 57, 95], np.uint8),
        "upper": np.array([65, 217, 208], np.uint8),
    },
    "blue": {
        "lower": np.array([82, 79, 189], np.uint8),
        "upper": np.array([110, 255, 255], np.uint8),
    },
    "white": {
        "lower": np.array([0, 0, 200], np.uint8),
        "upper": np.array([180, 255, 255], np.uint8),
    },
    "green_circle": {
        "lower": np.array([50, 30, 128], np.uint8),
        "upper": np.array([87, 255, 255], np.uint8),
    },
    "blue_circle": {
        "lower": np.array([92, 71, 88], np.uint8),
        "upper": np.array([122, 255, 255], np.uint8),
    },
    "up_green": {
        "lower": np.array([58, 89, 130], np.uint8),
        "upper": np.array([79, 255, 255], np.uint8),
    },
    "all_color": {
        "lower": np.array([0, 84, 179], np.uint8),
        "upper": np.array([180, 255, 255], np.uint8),
    },
}

kernel = np.ones((3, 3), np.uint8)

# 打开摄像头并设置参数
def open_camera():
    # Windows优先使用CAP_DSHOW，Linux使用CAP_V4L2
    if sys.platform.startswith("win"):
        cap = cv.VideoCapture(CAMERA_INDEX, cv.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv.VideoCapture(CAMERA_INDEX, cv.CAP_ANY)
        if not cap.isOpened():
            return cap
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc(*"MJPG"))
    else:
        cap = cv.VideoCapture(CAMERA_INDEX, cv.CAP_V4L2)
        if not cap.isOpened():
            return cap

    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv.CAP_PROP_FPS, FPS)
    time.sleep(CAMERA_WARMUP_SECONDS)

    ok, frame = cap.read()
    if ok and frame is not None and frame.size > 0:
        print(f"Camera opened: index={CAMERA_INDEX}, shape={frame.shape}")
    else:
        print("Camera opened but first frame invalid")
    return cap

# 打开串口
def open_serial():
    if OUTPUT_MODE != "serial":
        print("Output mode: print (serial disabled)")
        return None

    if serial is None:
        print("pyserial not installed, serial output disabled")
        return None

    port = SERIAL_PORT_WINDOWS if sys.platform.startswith("win") else SERIAL_PORT_LINUX
    try:
        return serial.Serial(
            port=port,
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=SERIAL_TIMEOUT,
        )
    except Exception as exc:
        print(f"Serial open failed: {exc}. Running in print mode")
        return None

# 视觉处理类
class VisionProcessor:
    def __init__(self, ser_conn):
        self.action_flag = 0
        self.running = True
        self.lock = threading.Lock()
        self.ser = ser_conn
        self.last_tx = None
        self.current_fps = 0.0
        self.last_frame_ts = time.perf_counter()
        self.metric_window_seconds = 2.0
        self.metric_window_start = time.perf_counter()
        self.metric_frame_count = 0
        self.metric_detect_count = 0
        self.last_detection_rate = 0.0

    def set_mode(self, mode):
        if 0 <= mode <= 9:
            with self.lock:
                self.action_flag = mode
            self._reset_detection_metrics()
            print(f"Mode changed to {mode}")

    def _reset_detection_metrics(self):
        self.metric_window_start = time.perf_counter()
        self.metric_frame_count = 0
        self.metric_detect_count = 0
        self.last_detection_rate = 0.0

    def _update_runtime_metrics(self, detected):
        now = time.perf_counter()
        dt = now - self.last_frame_ts
        if dt > 1e-6:
            instant_fps = 1.0 / dt
            if self.current_fps <= 0:
                self.current_fps = instant_fps
            else:
                self.current_fps = 0.85 * self.current_fps + 0.15 * instant_fps
        self.last_frame_ts = now

        self.metric_frame_count += 1
        if detected:
            self.metric_detect_count += 1

        if (now - self.metric_window_start) >= self.metric_window_seconds:
            self.last_detection_rate = self.metric_detect_count / max(self.metric_frame_count, 1)
            self.metric_window_start = now
            self.metric_frame_count = 0
            self.metric_detect_count = 0

    def _draw_status_overlay(self, frame, current_flag, mode_name, detected, mask_ratio, target):
        status_color = (0, 220, 0) if detected else (0, 0, 255)
        status_text = "DETECTED" if detected else "NOT DETECTED"

        overlay = frame.copy()
        cv.rectangle(overlay, (10, 10), (700, 150), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        cv.putText(frame, f"Mode {current_flag}: {mode_name}", (20, 38), cv.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv.putText(frame, f"Status: {status_text}", (20, 68), cv.FONT_HERSHEY_SIMPLEX, 0.72, status_color, 2)
        cv.putText(
            frame,
            f"FPS:{self.current_fps:.1f}  DetectRate(2s):{self.last_detection_rate * 100:.1f}%  Mask:{mask_ratio * 100:.1f}%",
            (20, 98),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        metric_line = "A:N/A  C:N/A  F:N/A"
        center_line = "Center:(N/A,N/A)"
        if target is not None:
            fill_ratio = target.get("fill_ratio")
            fill_text = f"{fill_ratio:.3f}" if fill_ratio is not None else "N/A"
            metric_line = f"A:{target['area']:.0f}  C:{target['circularity']:.3f}  F:{fill_text}"
            cx, cy = target["center"]
            center_line = f"Center:({cx},{cy})"

        cv.putText(frame, metric_line, (20, 126), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv.putText(frame, center_line, (430, 126), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # 获取颜色掩码
    def get_color_mask(self, hsv_img, color):
        cfg = COLOR_THRESHOLDS[color]
        # 支持双区间（如红色）和单区间（如绿色）的颜色阈值配置
        if "lower1" in cfg:
            bounds = [(cfg["lower1"], cfg["upper1"]), (cfg["lower2"], cfg["upper2"])]
        else:
            bounds = [(cfg["lower"], cfg["upper"])]

        # cv.inRange函数会返回一个二值图像，多个掩码通过位或操作合并成一个最终的掩码
        masks = [cv.inRange(hsv_img, lower, upper) for lower, upper in bounds]
        out = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        for m in masks:
            # cv.bitwise_or函数对两个图像进行按位或操作，得到一个新的图像，其中每个像素的值是两个输入图像对应像素值的按位或结果
            out = cv.bitwise_or(out, m)
        return out

    def process_mask(self, mask, target_type):
        rules = TARGET_RULES[target_type]
        eroded = cv.erode(mask, kernel, iterations=rules["erode_iter"])
        return cv.dilate(eroded, kernel, iterations=rules["dilate_iter"])

    def _contour_center(self, contour):
        m = cv.moments(contour)
        if m["m00"] == 0:
            return None
        return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])

    def detect_target(self, mask, target_type):
        rules = TARGET_RULES[target_type]
        retrieval = cv.RETR_CCOMP if target_type == "ring" else cv.RETR_EXTERNAL
        contours, hierarchy = cv.findContours(mask, retrieval, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = None
        best_score = -1.0
        for i, cnt in enumerate(contours):
            area = cv.contourArea(cnt)
            if not (rules["area_min"] < area < rules["area_max"]):
                continue

            peri = cv.arcLength(cnt, True)
            if peri <= 0:
                continue
            circularity = 4 * np.pi * area / (peri * peri)
            if circularity < rules["circularity_min"]:
                continue

            center = self._contour_center(cnt)
            if center is None:
                continue

            (x, y), radius = cv.minEnclosingCircle(cnt)
            fill_ratio = None
            if target_type == "ring":
                circle_area = np.pi * radius * radius if radius > 0 else 0
                fill_ratio = area / circle_area if circle_area > 0 else 1.0
                has_hole = bool(hierarchy is not None and hierarchy[0][i][2] != -1)
                if (not has_hole) and (fill_ratio > rules["fill_ratio_max"]):
                    continue

            score = area * circularity
            if score > best_score:
                best_score = score
                x, y, w, h = cv.boundingRect(cnt)
                best = {
                    "contour": cnt,
                    "center": center,
                    "radius": int(radius),
                    "bbox": (x, y, w, h),
                    "area": area,
                    "circularity": circularity,
                    "fill_ratio": fill_ratio,
                }

        return best

    def find_lines(self, mask):
        edges = cv.Canny(mask, 50, 150)
        lines = cv.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=40, maxLineGap=10)
        if lines is None or lines.size == 0:
            return None
        longest = max(lines, key=lambda x: np.linalg.norm(x[0][2:] - x[0][:2]))
        x1, y1, x2, y2 = longest[0]
        dx, dy = x2 - x1, y2 - y1
        return (x1, y1, x2, y2), (dy / dx if dx != 0 else float("inf"))

    def send_serial(self, data_type, *values):
        header = 0x5D
        processed_values = []

        if data_type == "circle":
            header = 0x5A
        elif data_type == "line":
            header = 0x51

        for v in values:
            if data_type == "line":
                v_int = int(round(v))
                v_int = int(np.clip(v_int, -32768, 32767))
                if v_int < 0:
                    v_int += 1 << 16
                processed_values.extend([(v_int >> 8) & 0xFF, v_int & 0xFF])
            else:
                processed_values.append(int(np.clip(round(v), 0, 255)))

        packet = [0x55, header, self.action_flag] + processed_values + [0xAA]

        if self.ser is None or OUTPUT_MODE == "print":
            if packet != self.last_tx:
                print("TX " + " ".join(f"{b:02X}" for b in packet))
                self.last_tx = packet
            return

        self.ser.write(bytes(packet))

    def process_frame(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        with self.lock:
            current_flag = self.action_flag

        mode_name = "OFF" if current_flag == 0 else f"Mode {current_flag}"
        detected = False
        mask_ratio = 0.0
        metric_target = None
        mask_to_show = np.zeros(hsv.shape[:2], dtype=np.uint8)

        if current_flag == 9:
            mode_name = "White Line + Green Ring"
            white_mask = self.get_color_mask(hsv, "white")
            line_data = self.find_lines(white_mask)
            if line_data is not None:
                (x1, y1, x2, y2), slope = line_data
                cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv.putText(frame, "White Line", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                self.send_serial("line", slope * 1000)
                detected = True

            ring_mask = self.get_color_mask(hsv, "green_circle")
            ring_mask = self.process_mask(ring_mask, "ring")
            ring = self.detect_target(ring_mask, "ring")
            mask_to_show = ring_mask
            mask_ratio = float(np.count_nonzero(ring_mask)) / ring_mask.size
            if ring is not None:
                cx, cy = ring["center"]
                cv.drawContours(frame, [ring["contour"]], -1, (0, 255, 0), 2)
                cv.circle(frame, (cx, cy), ring["radius"], (255, 0, 0), 2)
                cv.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
                x, y, w, h = ring["bbox"]
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                self.send_serial("circle", cx / SCALE_X, cy / SCALE_Y)
                detected = True
                metric_target = ring

        elif current_flag in MODE_CONFIG:
            mode = MODE_CONFIG[current_flag]
            mode_name = mode["name"]
            mask = self.get_color_mask(hsv, mode["color"])
            mask = self.process_mask(mask, mode["target"])
            mask_to_show = mask
            mask_ratio = float(np.count_nonzero(mask)) / mask.size

            target = self.detect_target(mask, mode["target"])
            if target is not None:
                cx, cy = target["center"]
                cv.drawContours(frame, [target["contour"]], -1, mode["draw"], 2)
                cv.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
                if mode["target"] == "ring":
                    cv.circle(frame, (cx, cy), target["radius"], mode["draw"], 2)
                    x, y, w, h = target["bbox"]
                    cv.rectangle(frame, (x, y), (x + w, y + h), mode["draw"], 2)
                cv.putText(
                    frame,
                    mode["name"],
                    (max(10, cx - 70), max(20, cy - 15)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    mode["draw"],
                    2,
                )
                self.send_serial("circle", cx / SCALE_X, cy / SCALE_Y)
                detected = True
                metric_target = target

        cv.imshow("mask", mask_to_show)
        self._update_runtime_metrics(detected)
        self._draw_status_overlay(frame, current_flag, mode_name, detected, mask_ratio, metric_target)

        return frame


def serial_monitor(processor, stop_event):
    if processor.ser is None:
        return

    while not stop_event.is_set():
        try:
            if processor.ser.in_waiting > 0:
                data = processor.ser.read(processor.ser.in_waiting)
                for byte in data:
                    if 0x30 <= byte <= 0x39:
                        processor.set_mode(byte - 0x30)
        except Exception as exc:
            print(f"Serial error: {exc}")
        time.sleep(0.01)


def main():
    cap = open_camera()
    ser_conn = open_serial()
    processor = VisionProcessor(ser_conn)
    stop_event = threading.Event()

    if not cap.isOpened():
        print("Camera open failed")
        return

    serial_thread = threading.Thread(target=serial_monitor, args=(processor, stop_event))
    serial_thread.start()

    try:
        while processor.running:
            ret, frame = cap.read()
            if not ret:
                print("Camera error")
                break
            if frame is None or frame.size == 0:
                continue

            processed = processor.process_frame(frame)
            cv.imshow("Result", processed)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if ord("0") <= key <= ord("9"):
                processor.set_mode(key - ord("0"))
    finally:
        processor.running = False
        stop_event.set()
        serial_thread.join()
        cap.release()
        cv.destroyAllWindows()
        if ser_conn is not None:
            ser_conn.close()
        print("System shutdown")


if __name__ == "__main__":
    main()
