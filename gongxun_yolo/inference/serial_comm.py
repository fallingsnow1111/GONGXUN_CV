"""
串口通信模块 — gongxun_yolo
串口协议与原方案完全一致:
    检测帧: [0x55] [0x5A] [mode] [int(cx/3.5)] [int(cy/2.5)] [0xAA]
    空检测: [0x55] [0x5A] [0]    [0]           [0]           [0xAA]
    线检测: [0x55] [0x51] [mode] [slope_high]  [slope_low]   [0xAA]
"""
import serial
import threading
import time


class SerialComm:
    X_SCALE = 3.5   # 800 / 3.5 ≈ 228，单字节可存
    Y_SCALE = 2.5   # 600 / 2.5 = 240，单字节可存

    START_BYTE     = 0x55
    HEADER_OBJECT  = 0x5A   # 目标检测帧头
    HEADER_LINE    = 0x51   # 线检测帧头
    END_BYTE       = 0xAA

    def __init__(self, port="/dev/ttyAMA0", baudrate=9600):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.001,
        )
        self._action_flag = 0
        self._lock = threading.Lock()
        self._monitor_thread = None

    def start_monitor(self):
        """启动后台线程，持续监听下位机发来的模式切换命令（ASCII '0'-'9'）"""
        self._monitor_thread = threading.Thread(
            target=self._serial_monitor, daemon=True
        )
        self._monitor_thread.start()

    def _serial_monitor(self):
        while self.ser.is_open:
            try:
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting)
                    for byte in data:
                        if 0x30 <= byte <= 0x39:   # ASCII '0'~'9'
                            with self._lock:
                                self._action_flag = byte - 0x30
                            print(f"[Serial] 模式切换 -> {byte - 0x30}")
            except Exception as e:
                print(f"[Serial] 读取异常: {e}")
            time.sleep(0.01)

    @property
    def mode(self):
        with self._lock:
            return self._action_flag

    def send_detection(self, mode: int, cx: float, cy: float):
        """发送目标检测帧: [0x55][0x5A][mode][x/3.5][y/2.5][0xAA]"""
        x_byte = int(cx / self.X_SCALE) & 0xFF
        y_byte = int(cy / self.Y_SCALE) & 0xFF
        self.ser.write(bytes([self.START_BYTE, self.HEADER_OBJECT,
                              mode & 0xFF, x_byte, y_byte, self.END_BYTE]))

    def send_idle(self):
        """发送空闲帧 / 未检测到目标帧: [0x55][0x5A][0][0][0][0xAA]"""
        self.ser.write(bytes([self.START_BYTE, self.HEADER_OBJECT,
                              0, 0, 0, self.END_BYTE]))

    def send_line_data(self, mode: int, slope_x1000: int):
        """发送白线检测帧（mode 9）: [0x55][0x51][mode][high][low][0xAA]"""
        val = int(slope_x1000) & 0xFFFF
        high_byte = (val >> 8) & 0xFF
        low_byte  = val & 0xFF
        self.ser.write(bytes([self.START_BYTE, self.HEADER_LINE,
                              mode & 0xFF, high_byte, low_byte, self.END_BYTE]))

    def close(self):
        if self.ser.is_open:
            self.ser.close()
