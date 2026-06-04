"""
将训练好的 YOLOv8 权重导出为 ONNX 格式，供树莓派部署

用法:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --weights runs/gongxun_train/weights/best.pt
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="导出 ONNX 模型")
    p.add_argument("--weights", default="../runs/gongxun_train/weights/best.pt")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--opset",   type=int, default=12,
                   help="ONNX opset，12 兼容性最广")
    return p.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        print(f"[Export] 权重文件不存在: {weights}")
        print("[Export] 请先运行 python scripts/train.py 完成训练")
        return

    model = YOLO(str(weights))
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,   # onnx-simplifier 简化计算图
        dynamic=False,   # 静态 shape，树莓派推理更快
        half=False,      # 树莓派 CPU 无 FP16 加速
    )

    onnx_path = weights.parent / (weights.stem + ".onnx")
    print(f"\n[Export] 导出完成: {onnx_path}")
    print(f"[Export] 将 {onnx_path} 复制到树莓派的 models/ 目录下")
    print("[Export] 树莓派运行:")
    print(f"         python -m inference.main --model models/{onnx_path.name}")


if __name__ == "__main__":
    main()
