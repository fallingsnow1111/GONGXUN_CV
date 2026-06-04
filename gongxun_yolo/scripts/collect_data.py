"""
摄像头数据采集工具 — 用于构建 YOLO 训练数据集

用法:
    # 手动采集（按 's' 保存当前帧）
    python scripts/collect_data.py --class-name red_block

    # 自动定时采集（每隔 1.5 秒保存一帧）
    python scripts/collect_data.py --class-name green_ring --auto --interval 1.5

    # 采集到验证集
    python scripts/collect_data.py --class-name blue_block --split val

采集完成后用 LabelImg 标注（YOLO 格式），标签文件放到 dataset/labels/train/
"""
import argparse
import cv2 as cv
import os
import time
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(description="摄像头数据采集")
    p.add_argument("--output",      default=None,
                   help="图片保存目录（默认按 split 自动确定）")
    p.add_argument("--split",       default="train", choices=["train", "val"],
                   help="保存到训练集还是验证集")
    p.add_argument("--camera",      type=int, default=0)
    p.add_argument("--width",       type=int, default=800)
    p.add_argument("--height",      type=int, default=600)
    p.add_argument("--class-name",  default="",
                   help="目标类别前缀，如 red_block（帮助整理文件名）")
    p.add_argument("--auto",        action="store_true",
                   help="自动定时采集")
    p.add_argument("--interval",    type=float, default=1.0,
                   help="自动采集间隔（秒）")
    p.add_argument("--max-images",  type=int,   default=500)
    return p.parse_args()


def main():
    args = parse_args()

    # 确定保存目录
    save_dir = args.output or os.path.join(
        os.path.dirname(__file__), "..", "dataset", "images", args.split)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv.VideoCapture(args.camera, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    prefix       = f"{args.class_name}_" if args.class_name else ""
    count        = 0
    last_capture = 0.0

    print(f"[Collect] 保存目录: {os.path.abspath(save_dir)}")
    if args.auto:
        print(f"[Collect] 自动采集模式，间隔 {args.interval}s，按 'q' 退出")
    else:
        print("[Collect] 手动模式：按 's' 保存，按 'q' 退出")

    while cap.isOpened() and count < args.max_images:
        ret, frame = cap.read()
        if not ret:
            break

        # 显示状态
        display = frame.copy()
        mode_text = "AUTO" if args.auto else "按's'保存"
        cv.putText(display, f"已保存: {count}  {mode_text}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv.putText(display, args.class_name or "(未设类别)", (10, 65),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv.imshow("Data Collection", display)

        key = cv.waitKey(1) & 0xFF
        now = time.time()

        save_now = (args.auto and now - last_capture >= args.interval) or \
                   (not args.auto and key == ord('s'))

        if save_now:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{prefix}{ts}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv.imwrite(filepath, frame)
            count       += 1
            last_capture = now
            print(f"[Collect] 已保存 {filepath}  ({count}/{args.max_images})")

        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    print(f"\n[Collect] 完成。共保存 {count} 张图片到 {os.path.abspath(save_dir)}")
    print("[Collect] 下一步：用 LabelImg 标注，标签放到 dataset/labels/train/")


if __name__ == "__main__":
    main()
