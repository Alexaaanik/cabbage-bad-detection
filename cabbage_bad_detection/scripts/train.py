import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "data.yaml"
RUNS = ROOT / "runs"
MODELS = ROOT / "models"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--device", default="mps")
    p.add_argument("--name", default="train")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(DATA),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=str(RUNS),
        name=args.name,
        plots=True,
    )

    metrics = model.val(data=str(DATA), device=args.device)

    MODELS.mkdir(parents=True, exist_ok=True)
    best = RUNS / args.name / "weights" / "best.pt"
    dest = MODELS / f"{args.name}_best.pt"
    if best.exists():
        shutil.copy(best, dest)
        print(f"weights -> {dest}")

    info = {
        "run_name": args.name,
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    mp = MODELS / f"{args.name}_metrics.json"
    mp.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
