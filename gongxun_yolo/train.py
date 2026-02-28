"""
工训赛 YOLOv8 训练脚本
======================
使用方法:
    python train.py                    # 默认训练
    python train.py --epochs 200       # 指定轮次
    python train.py --resume           # 断点续训

训练前准备:
    1. pip install ultralytics
    2. 准备 dataset/images/train/ 和 val/ 下的图片
    3. 准备 dataset/labels/train/ 和 val/ 下的标注（YOLO格式）
    4. 确认 dataset.yaml 中的类别配置正确

训练完成后:
    - 最佳模型会保存在 runs/detect/train/weights/best.pt
    - 把 best.pt 复制到 models/ 目录即可被主程序加载
"""

import argparse
import os
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("请先安装 ultralytics: pip install ultralytics")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="工训赛 YOLOv8 训练")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="预训练模型 (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--data", type=str, default="dataset.yaml",
                        help="数据集配置文件路径")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮次")
    parser.add_argument("--batch", type=int, default=16,
                        help="批大小")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="训练图像大小")
    parser.add_argument("--device", type=str, default="0",
                        help="设备 (0=GPU, cpu=CPU)")
    parser.add_argument("--resume", action="store_true",
                        help="断点续训")
    parser.add_argument("--workers", type=int, default=4,
                        help="数据加载线程数")
    parser.add_argument("--name", type=str, default="gongxun",
                        help="实验名称")
    return parser.parse_args()


def check_dataset(data_yaml):
    """检查数据集是否就绪"""
    base_dir = os.path.dirname(os.path.abspath(data_yaml))

    for split in ["train", "val"]:
        img_dir = os.path.join(base_dir, "dataset", "images", split)
        lbl_dir = os.path.join(base_dir, "dataset", "labels", split)

        if not os.path.exists(img_dir):
            print(f"[WARNING] 图像目录不存在: {img_dir}")
            return False
        if not os.path.exists(lbl_dir):
            print(f"[WARNING] 标注目录不存在: {lbl_dir}")
            return False

        img_count = len([f for f in os.listdir(img_dir)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        lbl_count = len([f for f in os.listdir(lbl_dir)
                         if f.lower().endswith('.txt')])

        print(f"[INFO] {split}: {img_count} 张图片, {lbl_count} 个标注文件")

        if img_count == 0:
            print(f"[WARNING] {split} 集没有图片！")
            return False

    return True


def main():
    args = parse_args()

    print("=" * 50)
    print("  工训赛 YOLOv8 训练")
    print("=" * 50)
    print(f"  模型: {args.model}")
    print(f"  数据集: {args.data}")
    print(f"  轮次: {args.epochs}")
    print(f"  批大小: {args.batch}")
    print(f"  图像大小: {args.imgsz}")
    print(f"  设备: {args.device}")
    print("=" * 50)

    # 检查数据集
    if not check_dataset(args.data):
        print("\n[ERROR] 数据集检查未通过，请准备好训练数据后重试")
        print("提示: 图片放到 dataset/images/train/ 和 val/")
        print("      标注放到 dataset/labels/train/ 和 val/")
        return

    # 加载模型
    if args.resume:
        # 断点续训: 从上次训练中恢复
        last_pt = Path(f"runs/detect/{args.name}/weights/last.pt")
        if last_pt.exists():
            model = YOLO(str(last_pt))
            print(f"[INFO] 从断点恢复: {last_pt}")
        else:
            print(f"[ERROR] 找不到 {last_pt}，无法断点续训")
            return
    else:
        model = YOLO(args.model)
        print(f"[INFO] 加载预训练模型: {args.model}")

    # 开始训练
    print("\n[INFO] 开始训练...\n")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        name=args.name,
        patience=20,          # 早停：20轮无提升就停止
        save=True,
        save_period=10,       # 每10轮保存一次
        plots=True,           # 生成训练曲线图
        verbose=True,
    )

    # 复制最佳模型到 models/ 目录
    best_pt = Path(f"runs/detect/{args.name}/weights/best.pt")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    if best_pt.exists():
        dest = models_dir / "best.pt"
        shutil.copy2(best_pt, dest)
        print(f"\n[SUCCESS] 最佳模型已复制到: {dest}")
        print(f"[INFO] 主程序 color_line_det_yolo.py 会自动加载此模型")
    else:
        print(f"\n[WARNING] 未找到 best.pt，请手动从训练结果中复制")

    print("\n训练完成！")
    print(f"训练结果目录: runs/detect/{args.name}/")


if __name__ == "__main__":
    main()
