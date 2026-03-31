import numpy as np
import cv2 as cv
import time
import serial
import threading

# 全局配置
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
FPS = 30
SERIAL_TIMEOUT = 0.001

# 初始化摄像头
cap = cv.VideoCapture(0, cv.CAP_V4L2)
cap.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv.CAP_PROP_FPS, FPS)

# 初始化串口
ser = serial.Serial(
    port="/dev/ttyAMA0",
    baudrate=9600,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=SERIAL_TIMEOUT
)

# 颜色阈值定义
COLOR_THRESHOLDS = {
    'red': {
        'lower1': np.array([0, 100, 100], np.uint8),
        'upper1': np.array([10, 255, 255], np.uint8),
        'lower2': np.array([160, 100, 100], np.uint8),
        'upper2': np.array([180, 255, 255], np.uint8)
    },
    'green': {
        'lower': np.array([62, 128, 104], np.uint8),
        'upper': np.array([90, 255, 255], np.uint8)
    },
    'blue': {
        'lower': np.array([82, 79, 189], np.uint8),
        'upper': np.array([110, 255, 255], np.uint8)
    },
    'white': {
        'lower': np.array([0, 0, 200], np.uint8),
        'upper': np.array([180, 255, 255], np.uint8)
    },
    "green_circle":{
        'lower': np.array([50, 30, 128], np.uint8),
        'upper': np.array([87, 255, 255], np.uint8)
    },
    "blue_circle":{
        'lower': np.array([92, 71, 88], np.uint8),
        'upper': np.array([122, 255, 255], np.uint8)
    },
    "up_green":{
        'lower': np.array([58, 89, 130], np.uint8),
        'upper': np.array([79, 255, 255], np.uint8)
    },
    "all_color":{
        'lower': np.array([0, 84, 179], np.uint8),
        'upper': np.array([180, 255, 255], np.uint8)
    }
}

# 形态学核
kernel = np.ones((3, 3), np.uint8)


class VisionProcessor:
    def __init__(self):
        self.action_flag = 0
        self.running = True
        self.lock = threading.Lock()

    def get_color_mask(self, hsv_img, color):
        if color == 'red':
            mask1 = cv.inRange(hsv_img, COLOR_THRESHOLDS['red']['lower1'], COLOR_THRESHOLDS['red']['upper1'])
            mask2 = cv.inRange(hsv_img, COLOR_THRESHOLDS['red']['lower2'], COLOR_THRESHOLDS['red']['upper2'])
            return mask1 + mask2
        return cv.inRange(hsv_img, COLOR_THRESHOLDS[color]['lower'], COLOR_THRESHOLDS[color]['upper'])

    def process_mask(self, mask):
        if self.action_flag ==5:
            eroded = cv.erode(mask, kernel, iterations=1)
            return cv.dilate(eroded, kernel, iterations=1)
        else:
            eroded = cv.erode(mask, kernel, iterations=1)
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
        if lines is None or lines.size ==0:
            return None
        longest = max(lines, key=lambda x: np.linalg.norm(x[0][2:] - x[0][:2]))
        x1, y1, x2, y2 = longest[0]
        dx, dy = x2 - x1, y2 - y1
        return (x1, y1, x2, y2), dy / dx if dx != 0 else float('inf')

    def send_serial(self, data_type, *values):
        header = 0x5D  # 默认 header
        processed_values = []  # 初始化空列表，确保后续代码能访问
        if data_type == 'circle':
            header = 0x5A
        elif data_type == 'line':
            header = 0x51
        for v in values:
            if isinstance(v, int) and v >= 65536:
                raise ValueError("Value too large for 16-bit conversion!")
            if data_type == 'line':
                v=int(v)
                # 拆分成高 8 位和低 8 位
                high_byte = (v >> 8) & 0xFF
                low_byte = v & 0xFF
                processed_values.append(high_byte)
                processed_values.append(low_byte)
            else:
                processed_values.append(v)  # 其他情况直接发送

        data = bytes([0x55, header, self.action_flag] + processed_values + [0xAA])
        ser.write(data)

    def process_frame(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        with self.lock:
            current_flag = self.action_flag
        color_map = {
        1: ('red', (0, 0, 255)),    # 红色 (BGR格式)
        2: ('green', (0, 255, 0)),   # 绿色
        3: ('blue', (255, 0, 0)),    # 蓝色
        4: ('red', (0, 0, 255)),     # 红色圆环
        5: ('green', (0, 255, 0)),   # 绿色圆环
        6: ('blue', (255, 0, 0)),    # 蓝色圆环
        7: ('green', (0, 255, 0)),   # 特殊绿色
        8: ('all', (255, 255, 255)), # 所有颜色(白色)
        9: ('line', (0, 255, 255))   # 直线(黄色)
       }
        # 根据模式处理并可视化
        if current_flag == 9:  # 直线+圆环模式
            # 白色直线检测
            white_mask = self.get_color_mask(hsv, 'white')
            if line_data := self.find_lines(white_mask):
                (x1, y1, x2, y2), slope = line_data
                cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv.putText(frame, f"White Line", (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                self.send_serial('line', slope * 1000)

            # 绿色圆环检测
            green_mask = self.get_color_mask(hsv, 'green')
            processed_mask = self.process_mask(green_mask)
            if circle_data := self.find_circles(processed_mask):
                center, radius = circle_data
                cv.circle(frame, center, radius, (255, 0, 0), 2)
                cv.circle(frame, center, 2, (0, 255, 0), 3)
                cv.putText(frame, "Green Circle", (center[0] - 50, center[1] - radius - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                self.send_serial('circle', center[0] / 3.5, center[1] / 2.5)

        elif current_flag in [1, 2, 3, 4, 5, 6, 7, 8]:  # 颜色检测模式
            color_names = ['red', 'green', 'blue', 'red', 'green_circle', 'blue_circle', 'up_green', 'all_color']
            color_bgr = [(0, 0, 255), (0, 255, 0), (255, 0, 0),  # 红绿蓝
                         (0, 0, 255), (0, 255, 0), (255, 0, 0),
                         (0, 255, 0), (255, 255, 255)]  # all用白色

            color = color_names[current_flag - 1]
            mask = self.get_color_mask(hsv, color)
            processed_mask = self.process_mask(mask)
            cv.imshow('1', processed_mask)
            if current_flag in color_map:
                color_name, draw_color = color_map[current_flag]

            # 在原图上绘制检测结果
            contours, _ = cv.findContours(processed_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
             
            for i, cnt in enumerate(contours):
                area = cv.contourArea(cnt)
                if 5000 < area < 50000:  # 面积过滤
                    perimeter = cv.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.5:  # 只处理足够圆的形状
                            # 计算轮廓中心点
                            M = cv.moments(cnt)
                            if M["m00"] != 0:
                                cX = int(M["m10"] / M["m00"])
                                cY = int(M["m01"] / M["m00"])
                                # 在原始图像上绘制
                                cv.drawContours(frame, [cnt], -1, draw_color, 2)  # 绘制轮廓
                                # 绘制中心点标记（十字+圆点）
                                cv.circle(frame, (cX, cY), 5, (0, 0, 0), -1) 
                                cx_send=cX / 3.5
                                cy_send=cY / 2.5
                                self.send_serial('circle', int(cx_send), int(cy_send))
                                print(f"x:{int(cx_send)},y:{int(cy_send)}")
        return frame


def serial_monitor(processor):
    while getattr(threading.current_thread(), "running", True):
        try:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                for byte in data:
                    if 0x30 <= byte <= 0x39:  # '0'-'9'
                        with processor.lock:
                            processor.action_flag = byte - 0x30
                        print(f"Mode changed to {processor.action_flag}")
        except Exception as e:
            print(f"Serial error: {e}")
        time.sleep(0.01)


def main():
    processor = VisionProcessor()
    serial_thread = threading.Thread(target=serial_monitor, args=(processor,))
    serial_thread.running = True
    serial_thread.start()

    try:
        while processor.running:
            ret, frame = cap.read()
            if not ret:
                print("Camera error")
                break

            processed = processor.process_frame(frame)
            cv.imshow('Result', processed)

            if cv.waitKey(1) == ord('q'):
                break
    finally:
        processor.running = False
        serial_thread.running = False
        serial_thread.join()
        cap.release()
        cv.destroyAllWindows()
        ser.close()
        print("System shutdown")


if __name__ == "__main__":
    main()
