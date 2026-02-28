"""
YOLO 检测模块 - action 1~8 的目标检测
"""
import os
import cv2 as cv

from config import (
    MODEL_PATH, YOLO_CONF, YOLO_IOU, YOLO_IMGSZ,
    CLASS_NAMES, ACTION_TO_CLASS, ACTION_DRAW_COLOR,
    AREA_MIN, AREA_MAX, CX_SCALE, CY_SCALE,
)

# 延迟导入 ultralytics，避免未安装时整个模块崩溃
_YOLO_CLS = None


def _get_yolo_cls():
    global _YOLO_CLS
    if _YOLO_CLS is None:
        try:
            from ultralytics import YOLO
            _YOLO_CLS = YOLO
        except ImportError:
            print("[WARNING] ultralytics 未安装，请运行: pip install ultralytics")
    return _YOLO_CLS


def load_model():
    """加载 YOLO 模型，返回 model 或 None"""
    YOLO = _get_yolo_cls()
    if YOLO is None:
        return None
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] 模型文件不存在: {MODEL_PATH}")
        print("[WARNING] 请将训练好的 best.pt 放入 models/ 目录")
        return None
    try:
        model = YOLO(MODEL_PATH)
        print(f"[INFO] YOLO 模型加载成功: {MODEL_PATH}")
        print(f"[INFO] 模型类别: {model.names}")
        return model
    except Exception as e:
        print(f"[ERROR] YOLO 模型加载失败: {e}")
        return None


def detect(model, frame, action_flag, serial_comm):
    """
    用 YOLOv8 检测目标并发送结果。

    Args:
        model:       YOLO 模型实例
        frame:       BGR 图像
        action_flag: 当前模式 (1-8)
        serial_comm: SerialComm 实例

    Returns:
        标注后的 frame
    """
    target_classes = ACTION_TO_CLASS.get(action_flag, [])
    draw_color = ACTION_DRAW_COLOR.get(action_flag, (255, 255, 255))

    results = model.predict(
        frame,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        verbose=False,
        classes=target_classes,
    )

    best_box = None
    best_area = 0

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            if not (AREA_MIN < area < AREA_MAX):
                continue

            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

            # 画框 + 标签
            cv.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
            cv.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)

            if area > best_area:
                best_area = area
                best_box = ((x1 + x2) // 2, (y1 + y2) // 2, area, cls_name, conf)

    # 发送最大目标
    if best_box is not None:
        cx, cy, area, cls_name, conf = best_box
        cv.circle(frame, (cx, cy), 5, (0, 0, 0), -1)
        cv.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

        cx_send = int(cx / CX_SCALE)
        cy_send = int(cy / CY_SCALE)
        serial_comm.send_circle(action_flag, cx_send, cy_send)
        print(f"[YOLO] flag={action_flag} {cls_name} conf={conf:.2f} "
              f"x={cx_send} y={cy_send} area={area}")
    else:
        serial_comm.send_empty(action_flag)

    return frame
