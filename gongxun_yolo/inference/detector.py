"""
YOLOv8 ONNX 推理引擎
优先使用 ONNX Runtime，不可用时自动回退到 OpenCV DNN。
"""
import numpy as np
import cv2 as cv


CLASS_NAMES = ["red_block", "green_block", "blue_block",
               "red_ring",  "green_ring",  "blue_ring"]


class Detection:
    """单个检测结果"""
    def __init__(self, class_id: int, confidence: float,
                 x1: float, y1: float, x2: float, y2: float):
        self.class_id   = class_id
        self.class_name = CLASS_NAMES[class_id]
        self.confidence = confidence
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    @property
    def center(self):
        """bbox 中心点，替代原方案的 cv.moments 质心"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def __repr__(self):
        cx, cy = self.center
        return (f"Detection({self.class_name} conf={self.confidence:.2f} "
                f"center=({cx:.0f},{cy:.0f}))")


class YOLODetector:
    INPUT_SIZE = 640

    def __init__(self, model_path: str, conf_thresh: float = 0.5,
                 iou_thresh: float = 0.45):
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        self._scale = 1.0
        self._dx = self._dy = 0
        self._orig_w = self._orig_h = 0
        self._load_model(model_path)

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _load_model(self, model_path: str):
        try:
            import onnxruntime as ort
            self._session    = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
            self._backend    = "onnxruntime"
            print(f"[YOLODetector] 后端: ONNX Runtime  模型: {model_path}")
        except ImportError:
            print("[YOLODetector] onnxruntime 未安装，回退至 OpenCV DNN")
            self._session    = cv.dnn.readNetFromONNX(model_path)
            self._backend    = "opencv_dnn"
            print(f"[YOLODetector] 后端: OpenCV DNN  模型: {model_path}")

    # ------------------------------------------------------------------
    # 预处理：letterbox resize + 归一化 + batch 维度
    # ------------------------------------------------------------------
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(self.INPUT_SIZE / w, self.INPUT_SIZE / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv.resize(frame, (nw, nh))

        canvas = np.full((self.INPUT_SIZE, self.INPUT_SIZE, 3), 114, dtype=np.uint8)
        dx, dy = (self.INPUT_SIZE - nw) // 2, (self.INPUT_SIZE - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized

        # 保存反变换参数
        self._scale  = scale
        self._dx, self._dy = dx, dy
        self._orig_w, self._orig_h = w, h

        # BGR -> RGB, HWC -> CHW, [0,255] -> [0,1], 加 batch 维
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(blob, axis=0)

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def _infer(self, blob: np.ndarray) -> np.ndarray:
        if self._backend == "onnxruntime":
            return self._session.run(None, {self._input_name: blob})[0]
        else:
            self._session.setInput(blob)
            return self._session.forward()

    # ------------------------------------------------------------------
    # 后处理：解析 YOLOv8 输出 -> Detection 列表
    # ------------------------------------------------------------------
    def _postprocess(self, raw: np.ndarray) -> list:
        # YOLOv8 输出形状: (1, 4+nc, N)
        output = raw[0].T      # (N, 4+nc)
        boxes_raw  = output[:, :4]
        scores_all = output[:, 4:]

        class_ids   = np.argmax(scores_all, axis=1)
        confidences = scores_all[np.arange(len(class_ids)), class_ids]

        keep = confidences > self.conf_thresh
        if not np.any(keep):
            return []

        boxes_raw   = boxes_raw[keep]
        class_ids   = class_ids[keep]
        confidences = confidences[keep]

        # cx,cy,w,h -> x1,y1,x2,y2 (letterbox 坐标系)
        cx, cy, bw, bh = (boxes_raw[:, i] for i in range(4))
        x1 = ((cx - bw / 2) - self._dx) / self._scale
        y1 = ((cy - bh / 2) - self._dy) / self._scale
        x2 = ((cx + bw / 2) - self._dx) / self._scale
        y2 = ((cy + bh / 2) - self._dy) / self._scale

        x1 = np.clip(x1, 0, self._orig_w)
        y1 = np.clip(y1, 0, self._orig_h)
        x2 = np.clip(x2, 0, self._orig_w)
        y2 = np.clip(y2, 0, self._orig_h)

        # NMS
        boxes_nms = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices   = cv.dnn.NMSBoxes(
            boxes_nms, confidences.tolist(), self.conf_thresh, self.iou_thresh)

        detections = []
        for i in indices:
            idx = int(i[0]) if isinstance(i, (list, np.ndarray)) else int(i)
            cid = int(class_ids[idx])
            if 0 <= cid < len(CLASS_NAMES):
                detections.append(Detection(
                    class_id=cid,
                    confidence=float(confidences[idx]),
                    x1=float(x1[idx]), y1=float(y1[idx]),
                    x2=float(x2[idx]), y2=float(y2[idx]),
                ))
        return detections

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list:
        """完整推理流程，返回 Detection 列表"""
        blob = self._preprocess(frame)
        raw  = self._infer(blob)
        return self._postprocess(raw)
