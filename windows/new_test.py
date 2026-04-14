import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np


# Camera defaults
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_WARMUP_SECONDS = 0.5

# HSV thresholds
COLOR_HSV: Dict[str, Dict[str, np.ndarray]] = {
    "red": {
        "lower1": np.array([0, 100, 100], np.uint8),
        "upper1": np.array([10, 255, 255], np.uint8),
        "lower2": np.array([160, 100, 100], np.uint8),
        "upper2": np.array([180, 255, 255], np.uint8),
    },
    "green": {
        "lower": np.array([50, 30, 128], np.uint8),
        "upper": np.array([87, 255, 255], np.uint8),
    },
    "blue": {
        "lower": np.array([92, 71, 88], np.uint8),
        "upper": np.array([122, 255, 255], np.uint8),
    },
}

MODE_MAP = {
    0: [],
    1: ["red"],
    2: ["green"],
    3: ["blue"],
    4: ["red", "green", "blue"],
}

DRAW_COLOR = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
}


@dataclass
class CircleCandidate:
    center: Tuple[float, float]
    radius: float
    area: float
    circularity: float
    weight: float


class CenterKalmanTracker:
    """4D state Kalman tracker: [x, y, vx, vy]."""

    def __init__(self, dt: float = 1.0, base_gate: float = 45.0, max_lost: int = 10):
        self.kf = cv.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.06
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.5
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.zeros((4, 1), dtype=np.float32)

        self.initialized = False
        self.base_gate = base_gate
        self.max_lost = max_lost
        self.lost_count = 0

    def _predict(self) -> Tuple[float, float]:
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def _current_gate(self) -> float:
        px = float(self.kf.errorCovPre[0, 0])
        py = float(self.kf.errorCovPre[1, 1])
        sigma = math.sqrt(max(px + py, 1e-6))
        return max(self.base_gate, 2.2 * sigma)

    def update(
        self, measurement: Optional[Tuple[float, float]]
    ) -> Tuple[Optional[Tuple[float, float]], bool, bool, float]:
        if not self.initialized:
            if measurement is None:
                return None, False, False, self.base_gate
            x, y = measurement
            self.kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
            self.initialized = True
            self.lost_count = 0
            return (x, y), True, False, self.base_gate

        pred_xy = self._predict()
        gate = self._current_gate()

        if measurement is None:
            self.lost_count += 1
            return pred_xy, False, False, gate

        mx, my = measurement
        dist = math.hypot(mx - pred_xy[0], my - pred_xy[1])
        if dist > gate and self.lost_count < self.max_lost:
            self.lost_count += 1
            return pred_xy, False, True, gate

        meas = np.array([[mx], [my]], dtype=np.float32)
        corr = self.kf.correct(meas)
        self.lost_count = 0
        return (float(corr[0, 0]), float(corr[1, 0])), True, False, gate


def open_camera(index: int, width: int, height: int, fps: int) -> cv.VideoCapture:
    if sys.platform.startswith("win"):
        cap = cv.VideoCapture(index, cv.CAP_DSHOW)
        backend = "CAP_DSHOW"
        if not cap.isOpened():
            cap = cv.VideoCapture(index, cv.CAP_ANY)
            backend = "CAP_ANY"
        if cap.isOpened():
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc(*"MJPG"))
    else:
        cap = cv.VideoCapture(index, cv.CAP_V4L2)
        backend = "CAP_V4L2"

    if not cap.isOpened():
        return cap

    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FPS, fps)
    time.sleep(CAMERA_WARMUP_SECONDS)

    ok = False
    frame = None
    for _ in range(10):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            break
        time.sleep(0.05)

    if ok and frame is not None:
        real_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        real_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        real_fps = cap.get(cv.CAP_PROP_FPS)
        print(
            f"Camera opened: index={index}, backend={backend}, "
            f"size={real_w}x{real_h}, fps={real_fps:.1f}"
        )
    else:
        cap.release()

    return cap


def get_color_mask(hsv_img: np.ndarray, color: str) -> np.ndarray:
    cfg = COLOR_HSV[color]
    if "lower1" in cfg:
        mask1 = cv.inRange(hsv_img, cfg["lower1"], cfg["upper1"])
        mask2 = cv.inRange(hsv_img, cfg["lower2"], cfg["upper2"])
        mask = cv.bitwise_or(mask1, mask2)
    else:
        mask = cv.inRange(hsv_img, cfg["lower"], cfg["upper"])

    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    return mask


def roi_box_around(
    center: Tuple[float, float],
    shape: Tuple[int, int],
    half_size: int,
) -> Optional[Tuple[int, int, int, int]]:
    h, w = shape
    cx, cy = int(round(center[0])), int(round(center[1]))
    x1 = max(0, cx - half_size)
    y1 = max(0, cy - half_size)
    x2 = min(w, cx + half_size)
    y2 = min(h, cy + half_size)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return x1, y1, x2, y2


def extract_contour_candidates(
    mask: np.ndarray,
    min_area: float,
    min_circularity: float,
    min_radius: float,
    offset: Tuple[int, int] = (0, 0),
) -> List[CircleCandidate]:
    contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    out: List[CircleCandidate] = []

    ox, oy = offset
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv.arcLength(cnt, True)
        if peri <= 1e-6:
            continue

        circularity = 4.0 * math.pi * area / (peri * peri)
        if circularity < min_circularity:
            continue

        (x, y), radius = cv.minEnclosingCircle(cnt)
        if radius < min_radius:
            continue

        cx = float(x + ox)
        cy = float(y + oy)
        # 任务目标是中心稳定，权重采用半径优先，大环中心更稳。
        weight = max(radius, 1.0)
        out.append(
            CircleCandidate(
                center=(cx, cy),
                radius=float(radius),
                area=float(area),
                circularity=float(circularity),
                weight=float(weight),
            )
        )

    return out


def cluster_by_center(
    candidates: Sequence[CircleCandidate],
    center_eps: float,
) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []

    for cand in candidates:
        cx, cy = cand.center
        assigned = False
        for cluster in clusters:
            ccx, ccy = cluster["center"]  # type: ignore[index]
            if math.hypot(cx - ccx, cy - ccy) <= center_eps:
                members: List[CircleCandidate] = cluster["members"]  # type: ignore[index]
                members.append(cand)
                w_sum = sum(m.weight for m in members)
                if w_sum <= 1e-6:
                    mx = float(sum(m.center[0] for m in members) / len(members))
                    my = float(sum(m.center[1] for m in members) / len(members))
                else:
                    mx = float(sum(m.center[0] * m.weight for m in members) / w_sum)
                    my = float(sum(m.center[1] * m.weight for m in members) / w_sum)
                cluster["center"] = (mx, my)
                cluster["weight_sum"] = w_sum
                assigned = True
                break

        if not assigned:
            clusters.append(
                {
                    "center": (cx, cy),
                    "members": [cand],
                    "weight_sum": cand.weight,
                }
            )

    clusters.sort(
        key=lambda c: (float(c["weight_sum"]), len(c["members"])),  # type: ignore[index]
        reverse=True,
    )
    return clusters


def weighted_center(members: Sequence[CircleCandidate]) -> Tuple[float, float]:
    w_sum = sum(m.weight for m in members)
    if w_sum <= 1e-6:
        x = float(sum(m.center[0] for m in members) / len(members))
        y = float(sum(m.center[1] for m in members) / len(members))
        return x, y
    x = float(sum(m.center[0] * m.weight for m in members) / w_sum)
    y = float(sum(m.center[1] * m.weight for m in members) / w_sum)
    return x, y


def detect_center_simple(
    mask: np.ndarray,
    hint_center: Optional[Tuple[float, float]],
    roi_half_size: int,
) -> Optional[Dict[str, Any]]:
    h, w = mask.shape[:2]
    all_candidates: List[CircleCandidate] = []
    roi_used = False
    roi_box = None

    if hint_center is not None:
        roi_box = roi_box_around(hint_center, (h, w), half_size=roi_half_size)
        if roi_box is not None:
            x1, y1, x2, y2 = roi_box
            crop = mask[y1:y2, x1:x2]
            all_candidates = extract_contour_candidates(
                crop,
                min_area=80.0,
                min_circularity=0.40,
                min_radius=3.0,
                offset=(x1, y1),
            )
            roi_used = True

    # ROI 找不到时回退全图，避免长期丢检。
    if len(all_candidates) < 2:
        all_candidates = extract_contour_candidates(
            mask,
            min_area=80.0,
            min_circularity=0.40,
            min_radius=3.0,
            offset=(0, 0),
        )
        if not all_candidates:
            return None

    center_eps = max(8.0, 0.02 * math.hypot(w, h))
    clusters = cluster_by_center(all_candidates, center_eps=center_eps)
    if not clusters:
        return None

    best_cluster = None
    best_score = -1e9
    for cluster in clusters:
        ccx, ccy = cluster["center"]  # type: ignore[index]
        weight_sum = float(cluster["weight_sum"])  # type: ignore[index]
        members: List[CircleCandidate] = cluster["members"]  # type: ignore[index]
        score = weight_sum + 2.5 * len(members)
        if hint_center is not None:
            score -= 0.30 * math.hypot(ccx - hint_center[0], ccy - hint_center[1])
        if score > best_score:
            best_score = score
            best_cluster = cluster

    if best_cluster is None:
        return None

    members = best_cluster["members"]  # type: ignore[index]
    cx, cy = weighted_center(members)
    radii = [m.radius for m in members]
    mean_radius = float(np.mean(radii)) if radii else 0.0

    return {
        "center": (cx, cy),
        "members": members,
        "cluster_size": len(members),
        "cluster_weight": float(best_cluster["weight_sum"]),
        "candidate_count": len(all_candidates),
        "mean_radius": mean_radius,
        "roi_used": roi_used,
        "roi_box": roi_box,
    }


def draw_detection(
    frame: np.ndarray,
    det: Dict[str, Any],
    color_name: str,
    tracker_xy: Optional[Tuple[float, float]],
    accepted_measurement: bool,
    gated_out: bool,
    gate: float,
) -> None:
    draw = DRAW_COLOR[color_name]
    cx, cy = det["center"]
    center_pt = (int(round(cx)), int(round(cy)))

    members: List[CircleCandidate] = det["members"]
    for m in members[:14]:
        cc = (int(round(m.center[0])), int(round(m.center[1])))
        cv.circle(frame, cc, int(round(m.radius)), draw, 1, cv.LINE_AA)

    cv.circle(frame, center_pt, 5, (0, 255, 255), -1)
    if det["mean_radius"] > 1.0:
        cv.circle(frame, center_pt, int(round(det["mean_radius"])), (255, 255, 255), 1, cv.LINE_AA)

    if tracker_xy is not None:
        tx, ty = tracker_xy
        track_pt = (int(round(tx)), int(round(ty)))
        cv.drawMarker(frame, track_pt, (255, 255, 0), cv.MARKER_CROSS, 16, 2)
        cv.line(frame, center_pt, track_pt, (255, 255, 0), 1, cv.LINE_AA)

    state_text = "KF:MEAS" if accepted_measurement else "KF:PRED"
    if gated_out:
        state_text += "(GATED)"

    cv.putText(
        frame,
        (
            f"{color_name.upper()} center=({center_pt[0]},{center_pt[1]}) "
            f"cand={det['candidate_count']} cluster={det['cluster_size']}"
        ),
        (15, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.62,
        draw,
        2,
    )
    cv.putText(
        frame,
        (
            f"weight={det['cluster_weight']:.1f} meanR={det['mean_radius']:.1f} "
            f"roi={'Y' if det['roi_used'] else 'N'}"
        ),
        (15, 54),
        cv.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
    )
    cv.putText(
        frame,
        f"{state_text} gate={gate:.1f}px",
        (15, 80),
        cv.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 0),
        2,
    )

    roi_box = det["roi_box"]
    if roi_box is not None:
        x1, y1, x2, y2 = roi_box
        cv.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple center fusion detector for ring marker")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument("--fps", type=int, default=CAMERA_FPS)
    parser.add_argument("--roi-half-size", type=int, default=140)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cap = open_camera(args.camera_index, args.width, args.height, args.fps)
    if not cap.isOpened():
        print("Camera open failed. Try: python windows/find_camera_id.py")
        return

    tracker = CenterKalmanTracker(dt=1.0, base_gate=45.0, max_lost=10)
    mode = 4
    last_tracker_xy: Optional[Tuple[float, float]] = None

    last_ts = time.perf_counter()
    fps_smooth = 0.0

    print("Keys: 0=off, 1=red, 2=green, 3=blue, 4=all colors, q=quit")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Frame read failed")
            break

        now = time.perf_counter()
        dt = max(now - last_ts, 1e-6)
        last_ts = now
        inst_fps = 1.0 / dt
        fps_smooth = inst_fps if fps_smooth <= 0 else 0.85 * fps_smooth + 0.15 * inst_fps

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        active_colors = MODE_MAP.get(mode, [])
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        best_det = None
        best_color = None
        best_score = -1e9

        for color in active_colors:
            color_mask = get_color_mask(hsv, color)
            combined_mask = cv.bitwise_or(combined_mask, color_mask)

            det = detect_center_simple(
                color_mask,
                hint_center=last_tracker_xy,
                roi_half_size=args.roi_half_size,
            )
            if det is None:
                continue

            score = det["cluster_weight"] + 2.0 * det["cluster_size"]
            if score > best_score:
                best_score = score
                best_det = det
                best_color = color

        measurement = None
        if best_det is not None:
            c = best_det["center"]
            measurement = (float(c[0]), float(c[1]))

        tracker_xy, accepted, gated, gate = tracker.update(measurement)

        if tracker_xy is not None:
            last_tracker_xy = tracker_xy
        if tracker.lost_count > tracker.max_lost:
            last_tracker_xy = None

        if best_det is not None and best_color is not None:
            draw_detection(frame, best_det, best_color, tracker_xy, accepted, gated, gate)
        else:
            cv.putText(
                frame,
                "No center candidate",
                (15, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 0, 255),
                2,
            )
            if tracker_xy is not None:
                tx, ty = tracker_xy
                cv.drawMarker(
                    frame,
                    (int(round(tx)), int(round(ty))),
                    (255, 255, 0),
                    cv.MARKER_CROSS,
                    16,
                    2,
                )

        mode_text = {
            0: "OFF",
            1: "RED",
            2: "GREEN",
            3: "BLUE",
            4: "ALL",
        }.get(mode, f"MODE {mode}")

        cv.putText(
            frame,
            f"Mode:{mode_text} FPS:{fps_smooth:.1f}",
            (15, frame.shape[0] - 18),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv.imshow("new_test_mask", combined_mask)
        cv.imshow("new_test_detection", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if ord("0") <= key <= ord("4"):
            mode = key - ord("0")
            print(f"Mode changed to {mode}")

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
