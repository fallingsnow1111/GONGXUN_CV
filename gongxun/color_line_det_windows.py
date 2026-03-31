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

# Global configuration
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
FPS = 30
CAMERA_INDEX = 0
WINDOWS_CAMERA_INDICES = [0]
ALLOW_WINDOWS_INDEX_0_FALLBACK = False
CAMERA_WARMUP_SECONDS = 0.5
LINUX_SCAN_MAX_INDEX = 5
MIN_VALID_FRAME_MEAN = 2.0
WINDOWS_PROBE_FRAMES = 8
MAX_BLACK_PIXEL_RATIO = 0.85
HALF_BLACK_DARK_MEAN = 5.0
HALF_BLACK_BRIGHT_MEAN = 20.0
WINDOWS_STRICT_FRAME_VALIDATION = False

OUTPUT_MODE = "print"  # "print" or "serial"
SERIAL_PORT_WINDOWS = "COM5"
SERIAL_PORT_LINUX = "/dev/ttyAMA0"
SERIAL_TIMEOUT = 0.001

# Color thresholds
COLOR_THRESHOLDS = {
    "red": {
        "lower1": np.array([0, 100, 100], np.uint8),
        "upper1": np.array([10, 255, 255], np.uint8),
        "lower2": np.array([160, 100, 100], np.uint8),
        "upper2": np.array([180, 255, 255], np.uint8),
    },
    "green": {
        "lower": np.array([62, 128, 104], np.uint8),
        "upper": np.array([90, 255, 255], np.uint8),
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

# Ring detection tuning params
RING_AREA_MIN = 2000
RING_AREA_MAX = 90000
RING_CIRCULARITY_MIN = 0.45
RING_FILL_RATIO_MAX = 0.82


def _read_valid_probe_frame(cap, probe_frames):
    last_frame = None
    last_reason = "no-frame"
    for _ in range(probe_frames):
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            last_reason = "empty-frame"
            time.sleep(0.03)
            continue

        last_frame = frame
        ok, reason = _is_frame_usable(frame)
        if ok:
            return True, frame, "ok"
        last_reason = reason
        time.sleep(0.03)

    return False, last_frame, last_reason


def _is_frame_usable(frame):
    if frame is None or frame.size == 0:
        return False, "empty"

    if not WINDOWS_STRICT_FRAME_VALIDATION:
        # Debug mode: avoid rejecting valid devices in low light.
        if int(frame.max()) == 0:
            return False, "all-zero"
        return True, "ok-relaxed"

    if len(frame.shape) == 3:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        gray = frame

    mean_val = float(gray.mean())
    if mean_val < MIN_VALID_FRAME_MEAN:
        return False, "too-dark"

    black_ratio = float(np.mean(gray < 8))
    if black_ratio > MAX_BLACK_PIXEL_RATIO:
        return False, "mostly-black"

    h, w = gray.shape[:2]
    if w >= 2:
        left_mean = float(gray[:, : w // 2].mean())
        right_mean = float(gray[:, w // 2 :].mean())
        darker = min(left_mean, right_mean)
        brighter = max(left_mean, right_mean)
        if darker < HALF_BLACK_DARK_MEAN and brighter > HALF_BLACK_BRIGHT_MEAN:
            return False, "half-black"

    return True, "ok"


def open_camera():
    if sys.platform.startswith("linux"):
        # Raspberry Pi stable path: force MJPG + fixed 640x480 and scan /dev/videoX.
        linux_paths = [f"/dev/video{CAMERA_INDEX}"]
        linux_paths.extend(
            [f"/dev/video{i}" for i in range(LINUX_SCAN_MAX_INDEX) if i != CAMERA_INDEX]
        )

        for path in linux_paths:
            cap = cv.VideoCapture(path, cv.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc(*"MJPG"))
            cap.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv.CAP_PROP_FPS, FPS)
            time.sleep(CAMERA_WARMUP_SECONDS)

            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                frame_mean = float(frame.mean())
                print(f"Probe {path}: frame.mean={frame_mean:.2f}")
                if frame_mean > MIN_VALID_FRAME_MEAN:
                    print(f"Using camera: {path} (MJPG, {CAMERA_WIDTH}x{CAMERA_HEIGHT}@{FPS})")
                    return cap

            cap.release()

        print("No valid Linux camera found")
        return cv.VideoCapture()

    # Windows UVC path: prioritize DSHOW + MJPG for HF899, then fallback.
    backend_candidates = [cv.CAP_DSHOW, cv.CAP_ANY, cv.CAP_MSMF]
    fourcc_candidates = ["MJPG", "YUY2", None]
    candidates = list(WINDOWS_CAMERA_INDICES)
    if ALLOW_WINDOWS_INDEX_0_FALLBACK and 0 not in candidates:
        candidates.append(0)

    for backend in backend_candidates:
        for idx in candidates:
            for fourcc_name in fourcc_candidates:
                cap = cv.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue

                if fourcc_name is not None:
                    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc(*fourcc_name))
                cap.set(cv.CAP_PROP_CONVERT_RGB, 1)
                cap.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                cap.set(cv.CAP_PROP_FPS, FPS)
                time.sleep(CAMERA_WARMUP_SECONDS)

                ok, frame, reason = _read_valid_probe_frame(cap, WINDOWS_PROBE_FRAMES)
                if ok and frame is not None:
                    frame_mean = float(frame.mean())
                    frame_std = float(frame.std())
                    print(
                        "Using camera: "
                        f"index={idx}, backend={backend}, fourcc={fourcc_name or 'default'}, "
                        f"mean={frame_mean:.2f}, std={frame_std:.2f}"
                    )
                    return cap

                print(
                    "Reject camera combo: "
                    f"index={idx}, backend={backend}, fourcc={fourcc_name or 'default'}, reason={reason}"
                )

                cap.release()

    print(
        "No valid Windows UVC camera found with current policy: "
        f"indices={candidates}, allow_index_0_fallback={ALLOW_WINDOWS_INDEX_0_FALLBACK}"
    )
    return cv.VideoCapture()


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


class VisionProcessor:
    def __init__(self, ser_conn):
        self.action_flag = 0
        self.running = True
        self.lock = threading.Lock()
        self.ser = ser_conn
        self.last_tx = None

    def get_color_mask(self, hsv_img, color):
        if color == "red":
            mask1 = cv.inRange(
                hsv_img,
                COLOR_THRESHOLDS["red"]["lower1"],
                COLOR_THRESHOLDS["red"]["upper1"],
            )
            mask2 = cv.inRange(
                hsv_img,
                COLOR_THRESHOLDS["red"]["lower2"],
                COLOR_THRESHOLDS["red"]["upper2"],
            )
            return mask1 + mask2
        return cv.inRange(hsv_img, COLOR_THRESHOLDS[color]["lower"], COLOR_THRESHOLDS[color]["upper"])

    def process_mask(self, mask):
        eroded = cv.erode(mask, kernel, iterations=1)
        if self.action_flag == 5:
            return cv.dilate(eroded, kernel, iterations=1)
        return cv.dilate(eroded, kernel, iterations=4)

    def find_circles(self, mask):
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv.contourArea)
        area = cv.contourArea(largest)
        if not 2000 < area < 90000:
            return None

        (x, y), radius = cv.minEnclosingCircle(largest)
        return (int(x), int(y)), int(radius)

    def find_lines(self, mask):
        edges = cv.Canny(mask, 50, 150)
        lines = cv.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=40, maxLineGap=10)
        if lines is None or lines.size == 0:
            return None

        longest = max(lines, key=lambda x: np.linalg.norm(x[0][2:] - x[0][:2]))
        x1, y1, x2, y2 = longest[0]
        dx, dy = x2 - x1, y2 - y1
        return (x1, y1, x2, y2), (dy / dx if dx != 0 else float("inf"))

    def find_ring(self, mask):
        contours, hierarchy = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = None
        best_score = -1.0
        for idx, cnt in enumerate(contours):
            area = cv.contourArea(cnt)
            if not (RING_AREA_MIN < area < RING_AREA_MAX):
                continue

            perimeter = cv.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < RING_CIRCULARITY_MIN:
                continue

            (x, y), radius = cv.minEnclosingCircle(cnt)
            if radius <= 1.0:
                continue

            circle_area = np.pi * radius * radius
            fill_ratio = area / circle_area if circle_area > 0 else 1.0
            has_hole = bool(hierarchy is not None and hierarchy[0][idx][2] != -1)

            # Ring: has hole OR not close to a filled disk.
            if (not has_hole) and (fill_ratio > RING_FILL_RATIO_MAX):
                continue

            score = area * circularity
            if score > best_score:
                best_score = score
                best = {
                    "center": (int(x), int(y)),
                    "radius": int(radius),
                    "contour": cnt,
                    "fill_ratio": fill_ratio,
                }

        return best

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
                processed_values.append((v_int >> 8) & 0xFF)
                processed_values.append(v_int & 0xFF)
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

        if current_flag == 9:
            white_mask = self.get_color_mask(hsv, "white")
            line_data = self.find_lines(white_mask)
            if line_data is not None:
                (x1, y1, x2, y2), slope = line_data
                cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv.putText(frame, "White Line", (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                self.send_serial("line", slope * 1000)

            green_mask = self.get_color_mask(hsv, "green")
            processed_mask = self.process_mask(green_mask)
            circle_data = self.find_circles(processed_mask)
            if circle_data is not None:
                center, radius = circle_data
                cv.circle(frame, center, radius, (255, 0, 0), 2)
                cv.circle(frame, center, 2, (0, 255, 0), 3)
                cv.putText(
                    frame,
                    "Green Circle",
                    (center[0] - 50, center[1] - radius - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                self.send_serial("circle", center[0] / 3.5, center[1] / 2.5)

        elif current_flag in [1, 2, 3, 7, 8]:
            color_names = [
                "red",
                "green",
                "blue",
                "up_green",
                "all_color",
            ]
            color_bgr = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 0),
                (255, 255, 255),
            ]

            mode_to_idx = {1: 0, 2: 1, 3: 2, 7: 3, 8: 4}
            map_idx = mode_to_idx[current_flag]
            color = color_names[map_idx]
            draw_color = color_bgr[map_idx]
            mask = self.get_color_mask(hsv, color)
            processed_mask = self.process_mask(mask)
            cv.imshow("mask", processed_mask)

            contours, _ = cv.findContours(processed_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv.contourArea(cnt)
                if not (5000 < area < 50000):
                    continue

                perimeter = cv.arcLength(cnt, True)
                if perimeter <= 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity <= 0.5:
                    continue

                m = cv.moments(cnt)
                if m["m00"] == 0:
                    continue

                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                cv.drawContours(frame, [cnt], -1, draw_color, 2)
                cv.circle(frame, (cx, cy), 5, (0, 0, 0), -1)
                self.send_serial("circle", cx / 3.5, cy / 2.5)

        elif current_flag in [4, 5, 6]:
            ring_color_map = {
                4: ("red", (0, 0, 255), "Red Ring"),
                5: ("green_circle", (0, 255, 0), "Green Ring"),
                6: ("blue_circle", (255, 0, 0), "Blue Ring"),
            }
            color_key, draw_color, label = ring_color_map[current_flag]

            mask = self.get_color_mask(hsv, color_key)
            processed_mask = self.process_mask(mask)
            cv.imshow("mask", processed_mask)

            ring = self.find_ring(processed_mask)
            if ring is not None:
                center = ring["center"]
                radius = ring["radius"]
                cnt = ring["contour"]
                cv.drawContours(frame, [cnt], -1, draw_color, 2)
                cv.circle(frame, center, radius, draw_color, 2)
                cv.circle(frame, center, 3, (0, 255, 255), -1)
                cv.putText(
                    frame,
                    f"{label}",
                    (center[0] - 60, max(20, center[1] - radius - 10)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    draw_color,
                    2,
                )
                self.send_serial("circle", center[0] / 3.5, center[1] / 2.5)

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
                        with processor.lock:
                            processor.action_flag = byte - 0x30
                        print(f"Mode changed to {processor.action_flag}")
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
                with processor.lock:
                    processor.action_flag = key - ord("0")
                print(f"Mode changed to {processor.action_flag}")
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
