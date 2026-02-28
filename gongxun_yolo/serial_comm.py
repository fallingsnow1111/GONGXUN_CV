"""
串口通信模块 - 发送检测结果 & 接收模式切换指令
"""
import time
import threading
import serial

from config import (
    SERIAL_PORT, SERIAL_BAUDRATE, SERIAL_TIMEOUT, MODE_NAMES,
)


class SerialComm:
    """封装串口收发，线程安全"""

    def __init__(self):
        self.ser = None
        self._init_serial()

    def _init_serial(self):
        try:
            self.ser = serial.Serial(
                port=SERIAL_PORT,
                baudrate=SERIAL_BAUDRATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=SERIAL_TIMEOUT,
            )
            print("[INFO] 串口初始化成功")
        except serial.SerialException as e:
            print(f"[WARNING] 串口打开失败: {e}")
            print("[WARNING] 将在无串口模式下运行（调试用）")
            self.ser = None

    @property
    def available(self):
        return self.ser is not None

    # ---- 发送 ----
    def send_circle(self, action_flag: int, cx: int, cy: int):
        """发送圆形目标坐标 [0x55, 0x5A, flag, cx, cy, 0xAA]"""
        self._write(bytes([0x55, 0x5A, action_flag, cx, cy, 0xAA]))

    def send_line(self, action_flag: int, slope_int: int):
        """发送直线斜率 [0x55, 0x51, flag, high, low, 0xAA]"""
        slope_int = int(slope_int)
        high = (slope_int >> 8) & 0xFF
        low = slope_int & 0xFF
        self._write(bytes([0x55, 0x51, action_flag, high, low, 0xAA]))

    def send_empty(self, action_flag: int):
        """发送空数据（无检测结果）"""
        self._write(bytes([0x55, 0x5A, action_flag, 0, 0, 0xAA]))

    def _write(self, data: bytes):
        if self.ser is not None:
            self.ser.write(data)

    # ---- 接收（后台线程） ----
    def start_listener(self, processor):
        """启动串口监听线程，自动更新 processor.action_flag"""
        t = threading.Thread(target=self._listen_loop, args=(processor,), daemon=True)
        t.start()
        return t

    def _listen_loop(self, processor):
        if not self.available:
            print("[INFO] 无串口，监听线程退出")
            return

        while processor.running:
            try:
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting)
                    for byte in data:
                        if 0x30 <= byte <= 0x39:
                            new_flag = byte - 0x30
                            with processor.lock:
                                old_flag = processor.action_flag
                                processor.action_flag = new_flag
                            if old_flag != new_flag:
                                name = MODE_NAMES.get(new_flag, "未知")
                                method = ("YOLO" if (1 <= new_flag <= 8 and processor.model)
                                          else ("传统CV" if new_flag == 9 else "HSV"))
                                print(f"[MODE] {old_flag} → {new_flag} ({name}) [{method}]")
            except Exception as e:
                print(f"[ERROR] 串口错误: {e}")
            time.sleep(0.01)

    def close(self):
        if self.ser:
            self.ser.close()
