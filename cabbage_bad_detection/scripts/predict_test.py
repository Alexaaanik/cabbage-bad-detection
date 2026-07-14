import argparse
import json
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models" / "full_run_best.pt"))
    p.add_argument("--source", default=str(ROOT.parent / "test_photo"))
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    src = Path(args.source)
    imgs = sorted(x for x in src.iterdir() if x.suffix.lower() in (".jpg", ".jpeg", ".png"))

    summary = []
    for img in imgs:
        res = model.predict(source=str(img), conf=args.conf, device=args.device, verbose=False)[0]
        out_img = out / f"{img.stem}_detected.jpg"
        Image.fromarray(res.plot()[..., ::-1]).save(out_img, quality=92)

        confs = [round(c, 3) for c in res.boxes.conf.tolist()] if len(res.boxes) else []
        summary.append({
            "source_image": img.name,
            "result_image": out_img.name,
            "detections": len(res.boxes),
            "confidences": confs,
        })
        print(f"{img.name}: {len(res.boxes)}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
