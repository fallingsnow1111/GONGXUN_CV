"""
gongxun_yolo 主入口 — 替代原 gongxun/color_line_det.py

用法:
    # 模拟检测（无需模型，验证串口/模式切换框架）
    python -m inference.main --dummy --show

    # 真实 YOLO 推理
    python -m inference.main --model models/yolov8n_gongxun.onnx --show

    # 树莓派无显示屏（去掉 --show）
    python -m inference.main --model models/yolov8n_gongxun.onnx

    # Windows 串口测试
    python -m inference.main --dummy --serial-port COM5 --show
"""
import argparse
import time
import sys
import cv2 as cv

from inference.serial_comm  import SerialComm
from inference.line_detector import WhiteLineDetector

# 模式 -> 允许的 YOLO 类别 ID 集合
# mode 7（特殊绿）和 mode 2（普通绿）同属 green_block(1)，YOLO 无需区分
MODE_TO_CLASS_IDS = {
    1: {0},                  # red_block
    2: {1},                  # green_block
    3: {2},                  # blue_block
    4: {3},                  # red_ring
    5: {4},                  # green_ring
    6: {5},                  # blue_ring
    7: {1},                  # green_block（特殊绿）
    8: {0, 1, 2, 3, 4, 5},  # 全色不筛选
}

CAM_W, CAM_H, CAM_FPS = 800, 600, 30


def parse_args():
    p = argparse.ArgumentParser(description="gongxun YOLOv8 视觉系统")
    p.add_argument("--model",       default="models/yolov8n_gongxun.onnx",
                   help="ONNX 模型路径")
    p.add_argument("--dummy",       action="store_true",
                   help="使用模拟检测（无需训练好的模型）")
    p.add_argument("--serial-port", default="/dev/ttyAMA0")
    p.add_argument("--baudrate",    type=int, default=9600)
    p.add_argument("--camera",      type=int, default=0)
    p.add_argument("--conf-thresh", type=float, default=0.50,
                   help="YOLO 置信度阈值")
    p.add_argument("--width",       type=int, default=CAM_W)
    p.add_argument("--height",      type=int, default=CAM_H)
    p.add_argument("--show",        action="store_true",
                   help="显示调试窗口（树莓派无屏时不加）")
    return p.parse_args()


def _filter(detections, mode):
    """筛选出当前模式关心的检测结果"""
    allowed = MODE_TO_CLASS_IDS.get(mode, set())
    return [d for d in detections if d.class_id in allowed]


def _best(detections):
    """取面积最大的目标（与原方案按轮廓面积降序取第一个一致）"""
    return max(detections, key=lambda d: d.area) if detections else None


def main():
    args = parse_args()

    # --- 初始化检测器 ---
    if args.dummy:
        from utils.dummy_model import DummyDetector
        detector   = DummyDetector(args.width, args.height)
        use_dummy  = True
        print("[Main] 模式: 模拟检测（--dummy）")
    else:
        from inference.detector import YOLODetector
        detector   = YOLODetector(args.model, conf_thresh=args.conf_thresh)
        use_dummy  = False

    # --- 初始化串口 ---
    serial_comm = SerialComm(port=args.serial_port, baudrate=args.baudrate)
    serial_comm.start_monitor()
    print(f"[Main] 串口监听启动: {args.serial_port} @ {args.baudrate}")

    # --- 初始化白线检测器 ---
    line_det = WhiteLineDetector()

    # --- 初始化摄像头（开一次，保持打开）---
    cap = cv.VideoCapture(args.camera, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv.CAP_PROP_FPS, CAM_FPS)
    if not cap.isOpened():
        print("[Main] 摄像头打开失败，退出")
        sys.exit(1)
    print(f"[Main] 摄像头已打开: {args.width}x{args.height}")

    # --- 主循环 ---
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[Main] 读取帧失败，重试…")
                time.sleep(0.05)
                continue

            mode = serial_comm.mode

            # ── MODE 0: 空闲 ──────────────────────────────────────────
            if mode == 0:
                serial_comm.send_idle()
                time.sleep(0.05)
                continue

            # ── MODE 9: 白线（Hough，不用 YOLO）─────────────────────
            elif mode == 9:
                result = line_det.detect(frame)
                if result is not None:
                    (x1, y1, x2, y2), slope = result
                    serial_comm.send_line_data(mode, int(slope * 1000))
                    if args.show:
                        cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv.putText(frame, f"slope={slope:.3f}", (10, 60),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    serial_comm.send_idle()

            # ── MODE 1-8: YOLO 目标检测 ───────────────────────────────
            elif mode in MODE_TO_CLASS_IDS:
                dets  = detector.detect(frame, mode=mode) if use_dummy \
                        else detector.detect(frame)
                dets  = _filter(dets, mode)
                best  = _best(dets)

                if best is not None:
                    cx, cy = best.center
                    serial_comm.send_detection(mode, cx, cy)
                    print(f"[Mode {mode}] {best.class_name} "
                          f"conf={best.confidence:.2f} "
                          f"center=({int(cx)},{int(cy)}) "
                          f"scaled=({int(cx/SerialComm.X_SCALE)},"
                          f"{int(cy/SerialComm.Y_SCALE)})")

                    if args.show:
                        cv.rectangle(frame,
                                     (int(best.x1), int(best.y1)),
                                     (int(best.x2), int(best.y2)),
                                     (0, 255, 0), 2)
                        cv.circle(frame, (int(cx), int(cy)), 6, (255, 0, 0), -1)
                        cv.putText(frame,
                                   f"{best.class_name} {best.confidence:.2f}",
                                   (int(best.x1), max(int(best.y1) - 8, 12)),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                else:
                    serial_comm.send_idle()

            # ── 调试显示 ──────────────────────────────────────────────
            if args.show:
                cv.putText(frame, f"Mode: {mode}", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv.imshow("gongxun_yolo", frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[Main] 用户中断")
    finally:
        cap.release()
        cv.destroyAllWindows()
        serial_comm.close()
        print("[Main] 已关闭")


if __name__ == "__main__":
    main()
