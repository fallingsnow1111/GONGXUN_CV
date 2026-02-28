"""
工训赛视觉检测 - YOLOv8 版本 主入口
====================================
模块结构:
  config.py         - 全局配置（阈值、参数、类别映射）
  serial_comm.py    - 串口通信（收发封装）
  detector_yolo.py  - YOLO 目标检测 (action 1-8)
  detector_hsv.py   - HSV 回退检测 (action 1-8, YOLO不可用时)
  detector_line.py  - 直线 + 圆环检测 (action 9)
  main.py           - 主循环（本文件）

action_flag:
  0: 空闲  |  1-3: 红/绿/蓝物块  |  4-6: 红/绿/蓝圆环
  7: 上层绿色  |  8: 全色  |  9: 直线+圆环
"""

import sys
import time
import threading
import cv2 as cv

from config import CAMERA_WIDTH, CAMERA_HEIGHT, FPS
from serial_comm import SerialComm
import detector_yolo
import detector_hsv
import detector_line


# ==================== 摄像头 ====================
def init_camera():
    cap = cv.VideoCapture(0, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        print("[ERROR] 摄像头打开失败！")
        sys.exit(1)
    print(f"[INFO] 摄像头初始化成功 ({CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {FPS}fps)")
    return cap


# ==================== 视觉处理器 ====================
class VisionProcessor:
    """轻量调度器：根据 action_flag 分发到不同检测模块"""

    def __init__(self, serial_comm, model=None):
        self.serial_comm = serial_comm
        self.model = model
        self.action_flag = 0
        self.running = True
        self.lock = threading.Lock()

    def process_frame(self, frame):
        with self.lock:
            flag = self.action_flag

        if flag == 0:
            self.serial_comm.send_empty(flag)
            cv.putText(frame, "IDLE (flag=0)", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        elif flag == 9:
            frame = detector_line.detect(frame, flag, self.serial_comm)

        elif 1 <= flag <= 8:
            if self.model is not None:
                frame = detector_yolo.detect(self.model, frame, flag, self.serial_comm)
            else:
                cv.putText(frame, "[HSV FALLBACK]", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                frame = detector_hsv.detect(frame, flag, self.serial_comm)
        else:
            cv.putText(frame, f"Unknown flag={flag}", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame


# ==================== 主函数 ====================
def main():
    print("=" * 50)
    print("  工训赛视觉检测 - YOLOv8 版本")
    print("=" * 50)

    cap = init_camera()
    serial_comm = SerialComm()
    model = detector_yolo.load_model()

    processor = VisionProcessor(serial_comm, model)

    # 启动串口监听
    serial_comm.start_listener(processor)

    print("[INFO] 主循环启动...")
    print(f"[INFO] 检测模式: {'YOLOv8' if model else 'HSV回退'} (1-8) + 传统CV (9)")
    print("[INFO] 按 'q' 退出 | 按 '0'-'9' 键盘切换模式（调试用）")

    fps_time = time.time()
    fps_count = 0

    try:
        while processor.running:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 摄像头读取失败")
                break

            processed = processor.process_frame(frame)

            # FPS
            fps_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps_val = fps_count / elapsed
                fps_count = 0
                fps_time = time.time()
            else:
                fps_val = fps_count / max(elapsed, 0.001)

            # OSD
            with processor.lock:
                flag = processor.action_flag
            mode_str = ("YOLO" if (1 <= flag <= 8 and model)
                        else ("HSV" if 1 <= flag <= 8
                              else ("LINE" if flag == 9 else "IDLE")))
            cv.putText(processed, f"FPS:{fps_val:.0f} Flag:{flag} [{mode_str}]",
                       (10, CAMERA_HEIGHT - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv.imshow('Result', processed)

            # 键盘调试
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif ord('0') <= key <= ord('9'):
                with processor.lock:
                    processor.action_flag = key - ord('0')
                print(f"[KEY] 模式切换到 {key - ord('0')}")

    finally:
        processor.running = False
        cap.release()
        cv.destroyAllWindows()
        serial_comm.close()
        print("[INFO] 系统关闭")


if __name__ == "__main__":
    main()
