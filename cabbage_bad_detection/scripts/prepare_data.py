import argparse
from pathlib import Path

SKIP = {"4", "49", "111", "141", "627"}

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "Кочаны разметка 700+-2"
OUT = Path(__file__).resolve().parents[1] / "data"


def collect(split):
    img_dir = DATASET / split / "images"
    lbl_dir = DATASET / split / "labels"
    ok, bad = [], []
    for img in sorted(img_dir.glob("*.jpg")):
        if img.stem in SKIP or not (lbl_dir / f"{img.stem}.txt").exists():
            bad.append(img)
        else:
            ok.append(img)
    return ok, bad


def save_paths(paths, fpath):
    lines = [str(Path("..") / ".." / p.relative_to(ROOT)) for p in paths]
    fpath.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nc", type=int, default=1)
    p.add_argument("--names", nargs="+", default=["bad"])
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    tr, tr_skip = collect("train")
    va, va_skip = collect("valid")

    save_paths(tr, OUT / "train.txt")
    save_paths(va, OUT / "val.txt")

    (OUT / "data.yaml").write_text(
        f"train: train.txt\nval: val.txt\nnc: {args.nc}\nnames: {args.names}\n"
    )

    print(f"train: {len(tr)} ok, skip {len(tr_skip)}")
    print(f"valid: {len(va)} ok, skip {len(va_skip)}")


if __name__ == "__main__":
    main()
