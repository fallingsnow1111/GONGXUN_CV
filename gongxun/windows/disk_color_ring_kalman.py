import time
import threading
import importlib
import sys

import cv2 as cv
import numpy as np

try:
    serial = importlib.import_module("serial")
except ImportError:
    serial = None


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS = 15
CAMERA_WARMUP_SECONDS = 0.5

# Camera selection:
# 1) Put your external USB camera index first.
# 2) By default we do not fall back to index 0 (usually laptop built-in camera).
PREFERRED_CAMERA_INDICES = [2]
ALLOW_INDEX_0_FALLBACK = False

OUTPUT_MODE = "print"  # "print" or "serial"
SERIAL_PORT = "/dev/ttyAMA0"
SERIAL_BAUDRATE = 9600
SERIAL_TIMEOUT = 0.001

SERIAL_HEADER = 0x5A
SERIAL_FRAME_HEAD = 0x55
SERIAL_FRAME_TAIL = 0xAA

SCALE_X = 3.5
SCALE_Y = 2.5

KERNEL = np.ones((3, 3), np.uint8)


MODE_CONFIG = {
    1: {"target": "block", "color": "red", "draw": (0, 0, 255), "name": "Red Block"},
    2: {"target": "block", "color": "green", "draw": (0, 255, 0), "name": "Green Block"},
    3: {"target": "block", "color": "blue", "draw": (255, 0, 0), "name": "Blue Block"},
    4: {"target": "ring", "color": "red", "draw": (0, 0, 255), "name": "Red Ring"},
    5: {"target": "ring", "color": "green", "draw": (0, 255, 0), "name": "Green Ring"},
    6: {"target": "ring", "color": "blue", "draw": (255, 0, 0), "name": "Blue Ring"},
}


TARGET_RULES = {
    "block": {
        "area_min": 5000,
        "area_max": 60000,
        "circularity_min": 0.45,
        "erode_iter": 1,
        "dilate_iter": 4,
    },
    "ring": {
        "area_min": 2000,
        "area_max": 90000,
        "circularity_min": 0.55,
        "erode_iter": 1,
        "dilate_iter": 2,
    },
}


COLOR_THRESHOLDS = {
    "red": {
        "block": [
            ((0, 100, 100), (10, 255, 255)),
            ((160, 100, 100), (180, 255, 255)),
        ],
        "ring": [
            ((0, 43, 140), (10, 255, 255)),
            ((156, 43, 46), (180, 255, 255)),
        ],
    },
    "green": {
        "block": [
            ((62, 128, 104), (90, 255, 255)),
        ],
        "ring": [
            ((50, 30, 128), (87, 255, 255)),
        ],
    },
    "blue": {
        "block": [
            ((82, 79, 189), (110, 255, 255)),
        ],
        "ring": [
            ((92, 71, 88), (122, 255, 255)),
        ],
    },
}


class PointKalmanTracker:
    def __init__(self, process_noise=1e-2, measure_noise=8.0, max_predict_frames=5):
        self.kf = cv.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.kf.transitionMatrix = np.eye(4, dtype=np.float32)
        self.kf.processNoiseCov = np.array(
            [
                [process_noise, 0, 0, 0],
                [0, process_noise, 0, 0],
                [0, 0, process_noise * 100, 0],
                [0, 0, 0, process_noise * 100],
            ],
            np.float32,
        )
        self.kf.measurementNoiseCov = np.array(
            [[measure_noise, 0], [0, measure_noise]], np.float32
        )
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.initialized = False
        self.missed_frames = 0
        self.max_predict_frames = max_predict_frames

    def _set_dt(self, dt):
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )

    def reset(self):
        self.initialized = False
        self.missed_frames = 0

    def update(self, point, dt):
        x, y = float(point[0]), float(point[1])
        self._set_dt(dt)

        if not self.initialized:
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True
            self.missed_frames = 0
            return int(x), int(y)

        self.kf.predict()
        corrected = self.kf.correct(np.array([[x], [y]], np.float32))
        self.missed_frames = 0
        return int(corrected[0][0]), int(corrected[1][0])

    def predict_only(self, dt):
        if not self.initialized:
            return None

        self._set_dt(dt)
        predicted = self.kf.predict()
        self.missed_frames += 1
        if self.missed_frames > self.max_predict_frames:
            self.reset()
            return None
        return int(predicted[0][0]), int(predicted[1][0])


class DiskColorRingDetector:
    def __init__(self):
        self.running = True
        self.action_flag = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.debug_mask_window = True
        self.output_mode = OUTPUT_MODE
        self.last_packet = None

        self.cap = self._open_camera()
        self.ser = self._open_serial()
        self.prev_time = time.time()

        self.trackers = {
            1: PointKalmanTracker(measure_noise=10.0),
            2: PointKalmanTracker(measure_noise=10.0),
            3: PointKalmanTracker(measure_noise=10.0),
            4: PointKalmanTracker(measure_noise=7.0),
            5: PointKalmanTracker(measure_noise=7.0),
            6: PointKalmanTracker(measure_noise=7.0),
        }

    def _open_camera(self):
        print(
            f"Camera policy: preferred={PREFERRED_CAMERA_INDICES}, "
            f"allow_index_0_fallback={ALLOW_INDEX_0_FALLBACK}"
        )
        backend = cv.CAP_DSHOW if sys.platform.startswith("win") else cv.CAP_V4L2
        backend_candidates = [backend, cv.CAP_ANY]

        camera_indices = list(PREFERRED_CAMERA_INDICES)
        if ALLOW_INDEX_0_FALLBACK and 0 not in camera_indices:
            camera_indices.append(0)

        for idx in camera_indices:
            for backend_item in backend_candidates:
                cap = cv.VideoCapture(idx, backend_item)
                if not cap.isOpened():
                    cap.release()
                    continue

                cap.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                cap.set(cv.CAP_PROP_FPS, FPS)

                # Let sensor/driver settle after basic parameter set.
                time.sleep(CAMERA_WARMUP_SECONDS)

                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    real_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                    real_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                    real_fps = cap.get(cv.CAP_PROP_FPS)
                    print(
                        "Camera opened: "
                        f"index={idx}, backend={backend_item}, "
                        f"res={real_w}x{real_h}, fps={real_fps:.1f}"
                    )
                    return cap

                cap.release()
                print(
                    f"Camera index={idx}, backend={backend_item} opened but first frame invalid"
                )

        print("No preferred external camera found")

        if ALLOW_INDEX_0_FALLBACK:
            cap = cv.VideoCapture(0)
            cap.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv.CAP_PROP_FPS, FPS)
            print("Fallback camera opened: index=0")
            return cap

        # Return unopened capture and let run() print a clear failure.
        return cv.VideoCapture()

    def _open_serial(self):
        if self.output_mode != "serial":
            print("Output mode: print (serial disabled)")
            return None

        if serial is None:
            print("pyserial not installed, serial output disabled")
            return None

        try:
            return serial.Serial(
                port=SERIAL_PORT,
                baudrate=SERIAL_BAUDRATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=SERIAL_TIMEOUT,
            )
        except Exception as exc:
            print(f"Serial open failed: {exc}. Running without serial output")
            return None

    def get_mode(self):
        with self.lock:
            return self.action_flag

    def set_mode(self, mode):
        if mode not in [0, 1, 2, 3, 4, 5, 6]:
            return
        with self.lock:
            self.action_flag = mode
        print(f"Mode changed to {mode}")

    def get_color_mask(self, hsv_img, color_name, target_type):
        ranges = COLOR_THRESHOLDS[color_name][target_type]
        mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv.bitwise_or(mask, cv.inRange(hsv_img, lower_np, upper_np))
        return mask

    def process_mask(self, mask, target_type):
        rules = TARGET_RULES[target_type]
        eroded = cv.erode(mask, KERNEL, iterations=rules["erode_iter"])
        return cv.dilate(eroded, KERNEL, iterations=rules["dilate_iter"])

    def _contour_center(self, contour):
        m = cv.moments(contour)
        if m["m00"] == 0:
            return None
        return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])

    def detect_block_center(self, mask):
        rules = TARGET_RULES["block"]
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        best_contour = None
        best_area = 0
        for cnt in contours:
            area = cv.contourArea(cnt)
            if not (rules["area_min"] < area < rules["area_max"]):
                continue
            peri = cv.arcLength(cnt, True)
            if peri == 0:
                continue
            circularity = 4 * np.pi * area / (peri * peri)
            if circularity < rules["circularity_min"]:
                continue
            if area > best_area:
                best_area = area
                best_contour = cnt

        if best_contour is None:
            return None, None

        center = self._contour_center(best_contour)
        return center, best_contour

    def detect_ring_center(self, mask):
        rules = TARGET_RULES["ring"]
        contours, hierarchy = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        best_contour = None
        best_score = -1
        for idx, cnt in enumerate(contours):
            area = cv.contourArea(cnt)
            if not (rules["area_min"] < area < rules["area_max"]):
                continue

            peri = cv.arcLength(cnt, True)
            if peri == 0:
                continue
            circularity = 4 * np.pi * area / (peri * peri)
            if circularity < rules["circularity_min"]:
                continue

            (x, y), radius = cv.minEnclosingCircle(cnt)
            if radius <= 1.0:
                continue

            circle_area = np.pi * radius * radius
            fill_ratio = area / circle_area

            has_hole = False
            if hierarchy is not None:
                has_hole = hierarchy[0][idx][2] != -1

            if not has_hole and fill_ratio > 0.82:
                continue

            score = area * circularity
            if score > best_score:
                best_score = score
                best_contour = cnt

        if best_contour is None:
            return None, None

        center = self._contour_center(best_contour)
        return center, best_contour

    def send_serial(self, action, center):
        if center is None:
            packet = [SERIAL_FRAME_HEAD, SERIAL_HEADER, 0, 0, 0, SERIAL_FRAME_TAIL]
        else:
            x_send = int(np.clip(center[0] / SCALE_X, 0, 255))
            y_send = int(np.clip(center[1] / SCALE_Y, 0, 255))
            packet = [SERIAL_FRAME_HEAD, SERIAL_HEADER, action, x_send, y_send, SERIAL_FRAME_TAIL]

        if self.output_mode == "print" or self.ser is None:
            if packet != self.last_packet:
                hex_frame = " ".join(f"{x:02X}" for x in packet)
                print(f"TX {hex_frame}")
                self.last_packet = packet
            return

        self.ser.write(bytes(packet))

    def process_frame(self, frame, dt):
        mode = self.get_mode()
        display = frame.copy()

        if mode not in MODE_CONFIG:
            self.send_serial(0, None)
            return display, None

        config = MODE_CONFIG[mode]
        target_type = config["target"]
        color_name = config["color"]
        draw_color = config["draw"]
        label = config["name"]

        smooth = cv.GaussianBlur(frame, (3, 3), 0)
        hsv = cv.cvtColor(smooth, cv.COLOR_BGR2HSV)
        mask = self.get_color_mask(hsv, color_name, target_type)
        filtered = self.process_mask(mask, target_type)

        if target_type == "ring":
            measured_center, best_contour = self.detect_ring_center(filtered)
        else:
            measured_center, best_contour = self.detect_block_center(filtered)

        tracker = self.trackers[mode]
        if measured_center is not None:
            kalman_center = tracker.update(measured_center, dt)
        else:
            kalman_center = tracker.predict_only(dt)

        if best_contour is not None:
            cv.drawContours(display, [best_contour], -1, draw_color, 2)

        if measured_center is not None:
            cv.circle(display, measured_center, 4, (0, 255, 255), -1)

        if kalman_center is not None:
            cv.circle(display, kalman_center, 6, draw_color, -1)
            cv.putText(
                display,
                f"{label} KF ({kalman_center[0]}, {kalman_center[1]})",
                (20, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.65,
                draw_color,
                2,
            )
            self.send_serial(mode, kalman_center)
        else:
            cv.putText(
                display,
                f"{label} searching...",
                (20, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )
            self.send_serial(0, None)

        return display, filtered

    def serial_monitor(self):
        if self.ser is None:
            return

        while not self.stop_event.is_set():
            try:
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting)
                    for byte in data:
                        if 0x30 <= byte <= 0x39:
                            self.set_mode(byte - 0x30)
            except Exception as exc:
                print(f"Serial monitor error: {exc}")
            time.sleep(0.01)

    def run(self):
        if not self.cap.isOpened():
            print("Camera open failed")
            return

        serial_thread = threading.Thread(target=self.serial_monitor)
        serial_thread.start()

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Camera read failed")
                    break
                if frame is None or frame.size == 0:
                    print("Camera returned empty frame, skipping")
                    continue

                now = time.time()
                dt = max(now - self.prev_time, 1.0 / FPS)
                self.prev_time = now

                result, mask = self.process_frame(frame, dt)
                cv.imshow("disk_detector_result", result)
                if self.debug_mask_window and mask is not None:
                    cv.imshow("disk_detector_mask", mask)

                key = cv.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if ord("0") <= key <= ord("6"):
                    self.set_mode(key - ord("0"))
        finally:
            self.running = False
            self.stop_event.set()
            serial_thread.join()
            self.cap.release()
            cv.destroyAllWindows()
            if self.ser is not None:
                self.ser.close()
            print("System shutdown")


def main():
    detector = DiskColorRingDetector()
    detector.run()


if __name__ == "__main__":
    main()
