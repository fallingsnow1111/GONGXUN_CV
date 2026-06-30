# maixcam_vision_with_ack.py
# MaixCAM Pro 视觉模块代码 (带 ACK 应答机制)
# 适配 STM32 通信协议，支持 50ms 内应答，幂等性处理
# 颜色识别改为调用 YOLOv5 模型 (训练ID: 287477)
# 色环识别调用 YOLOv5 模型 (训练ID: 287487)

from maix import camera, display, image, uart, pinmap, app, time, sys, err, nn, gpio
import gc

# =========================
# 串口自适应配置
# =========================
device_id = sys.device_id()
if device_id == "maixcam2":
    pin_function = {"A21": "UART4_TX", "A22": "UART4_RX"}
    device = "/dev/ttyS4"
else:
    pin_function = {"A16": "UART0_TX", "A17": "UART0_RX"}
    device = "/dev/ttyS0"

for pin, func in pin_function.items():
    err.check_raise(pinmap.set_pin_function(pin, func), f"Failed set pin{pin} function to {func}")

serial = uart.UART(device, 115200)
fill_light = None
try:
    err.check_raise(pinmap.set_pin_function("B3", "GPIOB3"), "Failed set fill light pin")
    fill_light = gpio.GPIO("GPIOB3", gpio.Mode.OUT)
    fill_light.value(1)
    print("Fill light LED on")
except Exception as e:
    print(f"Fill light LED init failed: {e}")
print(f"串口已打开：{device}，波特率 115200")

# =========================
# 基础参数配置
# =========================
IMG_W = 640
IMG_H = 480
CAM_FPS = 30
CAM_BUFF_NUM = 1
SHOW_IMAGE = True
DISPLAY_EVERY_N = 3
PRINT_FPS = True
GC_EVERY_N = 80

SX = IMG_W / 640.0
SY = IMG_H / 480.0

def scale_roi(x, y, w, h):
    return [int(x * SX), int(y * SY), int(w * SX), int(h * SY)]

ROI_MODE_1 = scale_roi(150, 200, 250, 250)
ROI_MODE_2 = scale_roi(100,   0, 400, 300)
ROI_MODE_3 = scale_roi(150,   0, 340, 300)
ROI_MODE_5 = ROI_MODE_3

# =========================
# LAB 颜色阈值 (保留用于巡线模式5)
# =========================
THRESH_RED   = [[0, 80,  40,  80,   10,  80]]
THRESH_GREEN = [[0, 80, -120, -10,    0,  50]]
THRESH_BLUE  = [[0, 60,   0, 127, -128, -35]]
THRESH_BLACK = [[0, 35, -25, 25, -25, 25]]

COLOR_THRESHOLDS = [THRESH_RED, THRESH_GREEN, THRESH_BLUE]
COLOR_NAMES = ["RED", "GREEN", "BLUE"]
RING_NAMES = ["RED_RING", "GREEN_RING", "BLUE_RING"]

MIN_AREA = int(3600 * SX * SY)
PIXELS_THRESHOLD = max(80, MIN_AREA // 2)
AREA_THRESHOLD = MIN_AREA
STABLE_DELTA = max(2, int(5 * SX))

DATA_STOP = bytes([0x88, 0x88, 0x88, 0x88, 0x88, 0x88])

MODE_NAMES = {
    0: "Idle",
    1: "Color",
    2: "ScanLoc",
    3: "Track",
    5: "LineFollow",
    6: "QRCode",
    7: "RingYOLO"
}

# =========================
# 摄像头初始化
# =========================
cam = camera.Camera(IMG_W, IMG_H, fps=CAM_FPS, buff_num=CAM_BUFF_NUM)
cam.skip_frames(20)

try:
    cam.awb_mode(camera.AwbMode.Manual)
    cam.set_wb_gain([0.134, 0.0625, 0.0625, 0.1239])
except Exception:
    pass

try:
    cam.saturation(60)
except Exception:
    pass

if SHOW_IMAGE:
    disp = display.Display()
else:
    disp = None

# =========================
# YOLOv5 模型加载
# =========================
# 色环检测模型 (287487)
detector = None
try:
    detector = nn.YOLOv5(model="/root/models/maixhub/287487/model_287487.mud")
    print(f"色环检测模型加载成功: 输入尺寸 {detector.input_width()}x{detector.input_height()}")
except Exception as e:
    print(f"色环检测模型加载失败: {e}")

# 颜色识别模型 (287477) —— 新增
color_detector = None
try:
    color_detector = nn.YOLOv5(model="/root/models/maixhub/287477/model_287477.mud")
    print(f"颜色识别模型加载成功: 输入尺寸 {color_detector.input_width()}x{color_detector.input_height()}")
    try:
        if hasattr(color_detector, 'labels') and color_detector.labels:
            print(f"颜色识别模型标签: {color_detector.labels}")
    except Exception:
        pass
except Exception as e:
    print(f"颜色识别模型加载失败: {e}")

# =========================
# 工具与协议发送函数
# =========================
def filter_ascii(text):
    if text is None: return ""
    return ''.join([c if ord(c) < 128 else '?' for c in str(text)])

class PacketParser:
    def __init__(self):
        self.in_pkg = False
        self.buf = []
        self.cmd = 0

    def update(self, data):
        if not data: return None
        latest = None
        for b in data:
            if b == 0xAA and not self.in_pkg:
                self.in_pkg = True
                self.buf = [b]
            elif b == 0xBB and self.in_pkg:
                self.buf.append(b)
                if len(self.buf) >= 3:
                    latest = self.buf[1]
                    self.cmd = latest
                self.in_pkg = False
                self.buf = []
            elif self.in_pkg:
                self.buf.append(b)
                if len(self.buf) > 16:
                    self.in_pkg = False
                    self.buf = []
        return latest

# --- ACK 应答发送 ---
def send_ack(cmd, status):
    """
    发送 ACK 应答帧: 0xCC CMD STATUS (3字节)
    必须在收到命令后立即调用 (50ms内)
    """
    serial.write(bytes([0xCC, cmd, status]))

def send_color(color_idx):
    serial.write(bytes([0x66, 0x66, color_idx + 1, 0x08]))

def send_stop():
    serial.write(DATA_STOP)

def send_xy10(x, y):
    x10 = int(x * 10)
    y10 = int(y * 10)
    x10 = max(-32768, min(32767, x10))
    y10 = max(-32768, min(32767, y10))
    x_h = (x10 >> 8) & 0xFF
    x_l = x10 & 0xFF
    y_h = (y10 >> 8) & 0xFF
    y_l = y10 & 0xFF
    serial.write(bytes([0x88, x_h, x_l, y_h, y_l, 0x88]))

def send_line_error(err):
    err = max(-32768, min(32767, err))
    e_h = (err >> 8) & 0xFF
    e_l = err & 0xFF
    serial.write(bytes([0x11, e_h, e_l, 0x22]))

def truncate_utf8(text, max_bytes=255):
    result = bytearray()
    for char in str(text):
        encoded = char.encode("utf-8")
        if len(result) + len(encoded) > max_bytes: break
        result.extend(encoded)
    return bytes(result)

def build_qr_packet(payload):
    payload_bytes = truncate_utf8(payload, 255)
    payload_len = len(payload_bytes)
    checksum = payload_len
    for b in payload_bytes:
        checksum ^= b
    return bytes([0x77, 0x77, payload_len]) + payload_bytes + bytes([checksum, 0x0D, 0x0A])

def send_qr_payload(payload):
    serial.write(build_qr_packet(payload))

# =========================
# 图像处理辅助函数
# =========================
def bbox_area(blob):
    return int(blob[2]) * int(blob[3])

def get_color_name(cidx):
    """安全获取颜色名称，优先使用 COLOR_NAMES，其次模型标签"""
    if cidx is None:
        return "None"
    if 0 <= cidx < len(COLOR_NAMES):
        return COLOR_NAMES[cidx]
    # 尝试使用模型标签
    if color_detector is not None:
        try:
            if hasattr(color_detector, 'labels') and color_detector.labels:
                return str(color_detector.labels[cidx])
        except Exception:
            pass
    return f"Color{cidx}"

# =========================
# 颜色识别 —— 改为调用 YOLOv5 模型 (287477)
# =========================
def find_largest_color_blob(img, roi):
    """
    使用 YOLOv5 颜色识别模型 (287477) 检测色块
    返回 (color_idx, blob_tuple) 或 (None, None)
    blob_tuple = (x, y, w, h) 在全图像坐标系中
    与原 LAB 阈值版本接口完全兼容
    """
    if color_detector is None:
        print("警告: 颜色识别模型未加载!")
        return None, None

    model_w = color_detector.input_width()
    model_h = color_detector.input_height()
    img_small = img.resize(model_w, model_h)
    objs = color_detector.detect(img_small, conf_th=0.5, iou_th=0.45)

    scale_x = IMG_W / model_w
    scale_y = IMG_H / model_h

    best_color, best_blob, best_area = None, None, 0

    for obj in objs:
        # 将模型坐标缩放回原图坐标
        box_x = obj.x * scale_x
        box_y = obj.y * scale_y
        box_w = obj.w * scale_x
        box_h = obj.h * scale_y

        # 计算检测框中心
        cx = box_x + box_w / 2
        cy = box_y + box_h / 2

        # 检查中心是否在 ROI 内
        if cx < roi[0] or cx > roi[0] + roi[2] or cy < roi[1] or cy > roi[1] + roi[3]:
            continue

        area = box_w * box_h
        if area > best_area:
            best_area = area
            best_color = obj.class_id
            best_blob = (int(box_x), int(box_y), int(box_w), int(box_h))

    return best_color, best_blob

def center_relative_to_roi(blob, roi):
    cx_global = int(blob[0]) + int(blob[2]) / 2.0
    cy_global = int(blob[1]) + int(blob[3]) / 2.0
    return cx_global - roi[0], cy_global - roi[1]

def draw_blob(img, blob, color=image.COLOR_GREEN):
    img.draw_rect(int(blob[0]), int(blob[1]), int(blob[2]), int(blob[3]), color)

def get_line_y_at_x(x1, y1, x2, y2, x):
    dx = x2 - x1
    if dx == 0: return y1
    return y1 + (y2 - y1) * (x - x1) / dx

def process_line_mode(img, roi):
    try:
        lines = img.get_regression(THRESH_BLACK, roi=roi, area_threshold=100)
    except TypeError:
        lines = img.get_regression(THRESH_BLACK, area_threshold=100)

    found = False
    for line in lines:
        x1, y1 = int(line.x1()), int(line.y1())
        x2, y2 = int(line.x2()), int(line.y2())
        left_x = roi[0]
        right_x = roi[0] + roi[2] - 1
        left_y = get_line_y_at_x(x1, y1, x2, y2, left_x)
        right_y = get_line_y_at_x(x1, y1, x2, y2, right_x)
        err = int(left_y - right_y)
        send_line_error(err)
        if SHOW_IMAGE:
            img.draw_line(x1, y1, x2, y2, image.COLOR_GREEN, 2)
            img.draw_string(2, 22, filter_ascii("line err:{}".format(err)), image.COLOR_GREEN)
        found = True
        break

    if not found:
        send_line_error(0)

def qrcode_payload_to_text(payload):
    if payload is None: return None
    if isinstance(payload, bytes):
        try: return payload.decode("utf-8")
        except Exception: return payload.decode("utf-8", "replace")
    return str(payload)

def find_qrcode_maixpy(img):
    try:
        qrcodes = img.find_qrcodes()
    except Exception as e:
        return None, None, []
    if not qrcodes: return None, None, []
    qr = qrcodes[0]
    try: payload = qr.payload()
    except Exception: payload = None
    return qrcode_payload_to_text(payload), qr, qrcodes

def draw_qrcode_maixpy(img, qrcodes, payload):
    for qr in qrcodes:
        try:
            corners = qr.corners()
            for i in range(4):
                p1 = corners[i]
                p2 = corners[(i + 1) % 4]
                img.draw_line(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), image.COLOR_RED, 2)
        except Exception:
            try:
                rect = qr.rect()
                img.draw_rect(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]), image.COLOR_RED)
            except Exception: pass

    if payload is None:
        img.draw_string(2, 22, "scan QR", image.COLOR_GREEN)
    else:
        preview = str(payload)[:24]
        img.draw_string(2, 22, "QR:" + filter_ascii(preview), image.COLOR_GREEN)

def process_qrcode_mode(img, now, draw=False):
    global last_qr_payload, qr_missing_since_ms, last_qr_send_ms
    payload, qr, qrcodes = find_qrcode_maixpy(img)

    if payload is not None:
        qr_missing_since_ms = 0
        if now - last_qr_send_ms >= 1000:
            send_qr_payload(payload)
            last_qr_send_ms = now
            last_qr_payload = payload
            print("QR sent:", payload)
    else:
        if qr_missing_since_ms == 0:
            qr_missing_since_ms = now
        elif now - qr_missing_since_ms >= QR_REARM_MS:
            last_qr_payload = None
            last_qr_send_ms = 0

    if draw:
        draw_qrcode_maixpy(img, qrcodes, payload)
    return payload

# =========================
# 状态变量
# =========================
parser = PacketParser()
mode_command = 1

scan_num = 0
x1 = y1 = x5 = y5 = 0
scan_mode = 0
first_enter = True
color_flag = 0
last_color_flag = 0
loc_start_ms = 0
loc_active = False

frame_id = 0
fps_count = 0
fps_t0 = time.ticks_ms()
last_report_ms = 0
REPORT_INTERVAL_MS = 60
QR_REARM_MS = 800
last_qr_payload = None
qr_missing_since_ms = 0
last_qr_send_ms = 0

# =========================
# 主循环
# =========================
while not app.need_exit():
    frame_id += 1
    fps_count += 1
    now = time.ticks_ms()

    # 读取串口指令
    data = serial.read()
    new_cmd = parser.update(data)

    if new_cmd is not None:
        SUPPORTED_CMDS = [0x00, 0x01, 0x02, 0x03, 0x05, 0x06, 0x07]

        if new_cmd in SUPPORTED_CMDS:
            send_ack(new_cmd, 0x00)  # 成功
        else:
            send_ack(new_cmd, 0x01)  # 不支持的命令

        if new_cmd in SUPPORTED_CMDS:
            previous_mode = mode_command
            mode_command = new_cmd

            # 模式切换时的状态重置
            if mode_command == 0:
                scan_mode = 0
                first_enter = True
                color_flag = 0
                last_color_flag = 0
                loc_active = False
                loc_start_ms = 0

            elif mode_command == 6 and previous_mode != 6:
                last_qr_payload = None
                qr_missing_since_ms = 0
                last_qr_send_ms = 0

    img = cam.read()
    mode_name = MODE_NAMES.get(mode_command, "Unknown")

    # 模式0：待机
    if mode_command == 0:
        if SHOW_IMAGE and frame_id % DISPLAY_EVERY_N == 0:
            img.draw_string(2, 2, "Mode0:" + mode_name, image.COLOR_GREEN)
            disp.show(img)
        time.sleep_ms(1)
        continue

    # 模式1：颜色识别 (改为调用 YOLOv5 模型 287477)
    if mode_command == 1:
        roi = ROI_MODE_1
        cidx, blob = find_largest_color_blob(img, roi)
        if cidx is not None and now - last_report_ms >= REPORT_INTERVAL_MS:
            send_color(cidx)
            last_report_ms = now
        if SHOW_IMAGE and frame_id % DISPLAY_EVERY_N == 0:
            img.draw_rect(roi[0], roi[1], roi[2], roi[3], image.COLOR_BLUE)
            img.draw_string(2, 2, "Mode1:" + mode_name, image.COLOR_WHITE)
            if blob is not None:
                draw_blob(img, blob)
                img.draw_string(2, 22, get_color_name(cidx), image.COLOR_GREEN)
            else:
                img.draw_string(2, 22, "No Color", image.COLOR_RED)
            disp.show(img)

    # 模式2：扫描定位 (颜色检测同样使用 YOLOv5 模型)
    elif mode_command == 2:
        roi = ROI_MODE_2
        cidx, blob = find_largest_color_blob(img, roi)
        found = blob is not None

        if scan_mode == 0:
            if now - last_report_ms >= REPORT_INTERVAL_MS:
                send_stop()
                last_report_ms = now
            if found:
                bx, by = int(blob[0]), int(blob[1])
                scan_num += 1
                if scan_num == 1:
                    x1, y1 = bx, by
                elif scan_num == 5:
                    x5, y5 = bx, by
                    if abs(x1 - x5) < STABLE_DELTA and abs(y1 - y5) < STABLE_DELTA:
                        color_flag = cidx
                        if first_enter:
                            first_enter = False
                            scan_mode = 0
                        else:
                            if color_flag != last_color_flag:
                                scan_mode = 1
                                loc_active = False
                        last_color_flag = color_flag
                    else:
                        scan_mode = 0
                    scan_num = 0
            else:
                scan_num = 0

        if scan_mode == 1 and found:
            if not loc_active:
                loc_start_ms = now
                loc_active = True
            cx, cy = center_relative_to_roi(blob, roi)
            if now - last_report_ms >= REPORT_INTERVAL_MS:
                send_xy10(cx, cy)
                last_report_ms = now
            if now - loc_start_ms >= 3000:
                scan_mode = 0
                loc_active = False
                scan_num = 0

        if SHOW_IMAGE and frame_id % DISPLAY_EVERY_N == 0:
            img.draw_rect(roi[0], roi[1], roi[2], roi[3], image.COLOR_BLUE)
            img.draw_string(2, 2, "Mode2:" + mode_name, image.COLOR_WHITE)
            if found:
                draw_blob(img, blob)
                cx, cy = center_relative_to_roi(blob, roi)
                img.draw_string(2, 22, "m2 {} x{} y{}".format(get_color_name(cidx), int(cx), int(cy)), image.COLOR_GREEN)
            disp.show(img)

    # 模式3：追踪 (颜色检测同样使用 YOLOv5 模型)
    elif mode_command == 3:
        roi = ROI_MODE_3
        cidx, blob = find_largest_color_blob(img, roi)
        if blob is None:
            if now - last_report_ms >= REPORT_INTERVAL_MS:
                send_stop()
                last_report_ms = now
        else:
            cx, cy = center_relative_to_roi(blob, roi)
            if now - last_report_ms >= REPORT_INTERVAL_MS:
                send_xy10(cx, cy)
                last_report_ms = now
        if SHOW_IMAGE and frame_id % DISPLAY_EVERY_N == 0:
            img.draw_rect(roi[0], roi[1], roi[2], roi[3], image.COLOR_BLUE)
            img.draw_string(2, 2, "Mode3:" + mode_name, image.COLOR_WHITE)
            if blob is not None:
                draw_blob(img, blob)
                img.draw_string(2, 22, "m3 {}".format(get_color_name(cidx)), image.COLOR_GREEN)
            disp.show(img)

    # 模式5：巡线
    elif mode_command == 5:
        process_line_mode(img, ROI_MODE_5)
        if SHOW_IMAGE and frame_id % DISPLAY_EVERY_N == 0:
            roi = ROI_MODE_5
            img.draw_rect(roi[0], roi[1], roi[2], roi[3], image.COLOR_BLUE)
            img.draw_string(2, 2, "Mode5:" + mode_name, image.COLOR_WHITE)
            disp.show(img)

    # 模式6：二维码
    elif mode_command == 6:
        draw_qr = SHOW_IMAGE and frame_id % DISPLAY_EVERY_N == 0
        process_qrcode_mode(img, now, draw=draw_qr)
        if draw_qr:
            img.draw_string(2, 2, "Mode6:" + mode_name, image.COLOR_WHITE)
            disp.show(img)

    # 模式7：YOLOv5 色环检测 (287487)
    elif mode_command == 7:
        if detector is None:
            img.draw_string(2, 2, "No Model!", image.COLOR_RED)
            if SHOW_IMAGE and frame_id % DISPLAY_EVERY_N == 0:
                disp.show(img)
            time.sleep_ms(100)
            continue

        model_w = detector.input_width()
        model_h = detector.input_height()
        img_small = img.resize(model_w, model_h)
        objs = detector.detect(img_small, conf_th=0.5, iou_th=0.45)

        best_obj = None
        max_score = 0
        for obj in objs:
            if obj.score > max_score:
                max_score = obj.score
                best_obj = obj

        scale_x = IMG_W / model_w
        scale_y = IMG_H / model_h

        if best_obj:
            cx = (best_obj.x + best_obj.w / 2) * scale_x
            cy = (best_obj.y + best_obj.h / 2) * scale_y
            box_x = best_obj.x * scale_x
            box_y = best_obj.y * scale_y
            box_w = best_obj.w * scale_x
            box_h = best_obj.h * scale_y

            if now - last_report_ms >= REPORT_INTERVAL_MS:
                send_color(best_obj.class_id)
                send_xy10(cx, cy)
                last_report_ms = now

            if SHOW_IMAGE and frame_id % DISPLAY_EVERY_N == 0:
                img.draw_rect(int(box_x), int(box_y), int(box_w), int(box_h), image.COLOR_RED, 2)
                label_str = f"ID:{best_obj.class_id}"
                if 0 <= best_obj.class_id < len(RING_NAMES):
                    label_str = RING_NAMES[best_obj.class_id]
                info = f"{label_str} {best_obj.score:.2f}"
                img.draw_string(int(box_x), int(box_y) - 20, filter_ascii(info), image.COLOR_RED)
                img.draw_string(2, 2, "Mode7:" + mode_name, image.COLOR_WHITE)
                disp.show(img)
        else:
            if now - last_report_ms >= REPORT_INTERVAL_MS:
                send_stop()
                last_report_ms = now
            if SHOW_IMAGE and frame_id % DISPLAY_EVERY_N == 0:
                img.draw_string(2, 2, "Mode7:" + mode_name + " No Target", image.COLOR_WHITE)
                disp.show(img)

    if PRINT_FPS and now - fps_t0 >= 1000:
        print("fps:", fps_count, "cmd:", mode_command)
        fps_count = 0
        fps_t0 = now

    if frame_id % GC_EVERY_N == 0:
        gc.collect()

    time.sleep_ms(1)
