"""
模拟检测器 — 无需训练好的模型即可测试完整框架
用法: python -m inference.main --dummy --show
"""
import random
from inference.detector import Detection, CLASS_NAMES


# 模式 -> 允许返回的类别 ID
MODE_CLASS_MAP = {
    1: [0],              # red_block
    2: [1],              # green_block
    3: [2],              # blue_block
    4: [3],              # red_ring
    5: [4],              # green_ring
    6: [5],              # blue_ring
    7: [1],              # green_block（特殊绿 = 同类）
    8: [0, 1, 2, 3, 4, 5],  # 全色
}


class DummyDetector:
    """
    根据当前模式随机生成合理的检测结果。
    帧中心附近随机 bbox，70% 概率有目标。
    """

    def __init__(self, frame_w: int = 800, frame_h: int = 600):
        self.frame_w = frame_w
        self.frame_h = frame_h

    def detect(self, frame, mode: int = None) -> list:
        if mode not in MODE_CLASS_MAP:
            return []
        if random.random() < 0.3:      # 30% 概率无目标
            return []

        class_id = random.choice(MODE_CLASS_MAP[mode])
        cx = self.frame_w // 2 + random.randint(-120, 120)
        cy = self.frame_h // 2 + random.randint(-90, 90)
        bw = random.randint(60, 160)
        bh = random.randint(60, 160)

        return [Detection(
            class_id=class_id,
            confidence=round(random.uniform(0.60, 0.95), 2),
            x1=cx - bw // 2, y1=cy - bh // 2,
            x2=cx + bw // 2, y2=cy + bh // 2,
        )]
