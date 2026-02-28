"""
全局配置 - 所有常量和参数集中管理
"""
import os
import numpy as np

# ==================== 摄像头 ====================
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
FPS = 30

# ==================== 串口 ====================
SERIAL_PORT = "/dev/ttyAMA0"
SERIAL_BAUDRATE = 9600
SERIAL_TIMEOUT = 0.001

# ==================== YOLO ====================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
YOLO_CONF = 0.5
YOLO_IOU = 0.45
YOLO_IMGSZ = 640

# 类别 ID → 名称（与 dataset.yaml 一致）
CLASS_NAMES = {
    0: "red_block",
    1: "green_block",
    2: "blue_block",
    3: "red_ring",
    4: "green_ring",
    5: "blue_ring",
}

# action_flag → 需要检测的 YOLO 类别 ID
ACTION_TO_CLASS = {
    1: [0],                       # 红色物块
    2: [1],                       # 绿色物块
    3: [2],                       # 蓝色物块
    4: [3],                       # 红色圆环
    5: [4],                       # 绿色圆环
    6: [5],                       # 蓝色圆环
    7: [1, 4],                    # 上层绿色
    8: [0, 1, 2, 3, 4, 5],       # 全色
}

# action_flag → 绘制颜色 (BGR)
ACTION_DRAW_COLOR = {
    1: (0, 0, 255),
    2: (0, 255, 0),
    3: (255, 0, 0),
    4: (0, 0, 255),
    5: (0, 255, 0),
    6: (255, 0, 0),
    7: (0, 255, 0),
    8: (255, 255, 255),
}

# action_flag → 模式名称
MODE_NAMES = {
    0: "空闲",   1: "红色物块", 2: "绿色物块",
    3: "蓝色物块", 4: "红色圆环", 5: "绿色圆环",
    6: "蓝色圆环", 7: "上层绿色", 8: "全色检测",
    9: "直线检测",
}

# ==================== 面积过滤 ====================
AREA_MIN = 2000
AREA_MAX = 90000

# ==================== 坐标映射比例 ====================
CX_SCALE = 3.5
CY_SCALE = 2.5

# ==================== HSV 阈值（回退 + 直线模式） ====================
HSV_THRESHOLDS = {
    'red': {
        'lower1': np.array([0, 100, 100], np.uint8),
        'upper1': np.array([10, 255, 255], np.uint8),
        'lower2': np.array([160, 100, 100], np.uint8),
        'upper2': np.array([180, 255, 255], np.uint8),
    },
    'green': {
        'lower': np.array([62, 128, 104], np.uint8),
        'upper': np.array([90, 255, 255], np.uint8),
    },
    'blue': {
        'lower': np.array([82, 79, 189], np.uint8),
        'upper': np.array([110, 255, 255], np.uint8),
    },
    'white': {
        'lower': np.array([0, 0, 200], np.uint8),
        'upper': np.array([180, 255, 255], np.uint8),
    },
    'green_circle': {
        'lower': np.array([50, 30, 128], np.uint8),
        'upper': np.array([87, 255, 255], np.uint8),
    },
    'blue_circle': {
        'lower': np.array([92, 71, 88], np.uint8),
        'upper': np.array([122, 255, 255], np.uint8),
    },
    'up_green': {
        'lower': np.array([58, 89, 130], np.uint8),
        'upper': np.array([79, 255, 255], np.uint8),
    },
    'all_color': {
        'lower': np.array([0, 84, 179], np.uint8),
        'upper': np.array([180, 255, 255], np.uint8),
    },
}

# action_flag → HSV 颜色键名
ACTION_TO_HSV_COLOR = {
    1: 'red',  2: 'green',  3: 'blue',
    4: 'red',  5: 'green_circle',  6: 'blue_circle',
    7: 'up_green',  8: 'all_color',
}

# 形态学核
KERNEL_3x3 = np.ones((3, 3), np.uint8)
