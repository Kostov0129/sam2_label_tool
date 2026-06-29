import argparse
import json
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Show probe training status from checkpoints.")
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    meta_path = run_dir / "train_meta.json"
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        epochs = meta.get("args", {}).get("epochs", "?")
        batch_size = meta.get("args", {}).get("batch_size", "?")
        classes = ",".join(meta.get("class_names", []))
        print(f"classes={classes} batch_size={batch_size} target_epochs={epochs}")
    else:
        epochs = "?"
        print("train_meta.json not found")

    for label, path in [("last", last_path), ("best", best_path)]:
        if not path.exists():
            print(f"{label}: missing")
            continue
        ckpt = torch.load(path, map_location="cpu")
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))
        epoch = ckpt.get("epoch", "?")
        val_iou = ckpt.get("val_iou", None)
        val_text = f"{val_iou:.4f}" if isinstance(val_iou, float) else str(val_iou)
        print(f"{label}: epoch={epoch}/{epochs} val_iou={val_text} updated={mtime}")


if __name__ == "__main__":
    main()
