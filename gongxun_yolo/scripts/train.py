"""
YOLOv8n 训练脚本 — 在 PC（GPU 机器）上运行，不在树莓派上运行

安装依赖:
    pip install ultralytics

用法:
    python scripts/train.py
    python scripts/train.py --epochs 200 --batch 32
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="训练 gongxun YOLOv8n")
    p.add_argument("--model",   default="yolov8n.pt",
                   help="基础模型（首次自动下载 COCO 预训练权重）")
    p.add_argument("--data",    default="../config/data.yaml")
    p.add_argument("--epochs",  type=int,   default=150)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--batch",   type=int,   default=16)
    p.add_argument("--device",  default="0", help="CUDA 设备编号，或 'cpu'")
    p.add_argument("--workers", type=int,   default=4)
    p.add_argument("--project", default="../runs")
    p.add_argument("--name",    default="gongxun_train")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)

    results = model.train(
        data     = args.data,
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = args.batch,
        device   = args.device,
        workers  = args.workers,
        project  = args.project,
        name     = args.name,

        # HSV 增强参数精调：
        # hue 偏移要小，防止 red_block 被增强成 green_block
        hsv_h=0.010,
        hsv_s=0.70,
        hsv_v=0.40,

        # 几何增强
        degrees=15.0,   # 机器人摄像头安装固定，适度旋转
        translate=0.10,
        scale=0.50,
        flipud=0.0,     # 不做垂直翻转（机器人视角固定）
        fliplr=0.5,

        # 其他
        mosaic=1.0,
        mixup=0.10,
        patience=30,    # 连续 30 epoch 无提升则提前停止
        save_period=10,
        val=True,
        plots=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n训练完成。最佳权重: {best}")
    print(f"下一步运行: python scripts/export_onnx.py --weights {best}")


if __name__ == "__main__":
    main()
