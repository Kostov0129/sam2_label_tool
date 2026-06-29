import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wp.probe_train.train_probe_sam2 import MODEL_IDS, PromptlessProbeSAM2


def overlay_mask(image_bgr, mask, color):
    out = image_bgr.copy()
    color_layer = np.zeros_like(out)
    color_layer[:, :] = color
    fg = mask > 0
    if np.any(fg):
        out[fg] = cv2.addWeighted(out[fg], 0.55, color_layer[fg], 0.45, 0)
    return out


def main():
    parser = argparse.ArgumentParser(description="Run promptless probe SAM2 segmentation.")
    parser.add_argument("--checkpoint", default="out/probe_sam2_train/best.pt")
    parser.add_argument("--input", required=True, help="Image file or directory.")
    parser.add_argument("--output_dir", default="out/probe_sam2_pred")
    parser.add_argument("--model_size", default=None, choices=[None, *sorted(MODEL_IDS)])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from sam2.sam2_image_predictor import SAM2ImagePredictor

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_size = args.model_size or ckpt.get("args", {}).get("model_size", "base")
    class_names = ckpt.get("class_names", ["probe_whole", "probe_yellow_side"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    predictor = SAM2ImagePredictor.from_pretrained(MODEL_IDS[model_size], device=str(device))
    model = PromptlessProbeSAM2(predictor.model, num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    input_path = Path(args.input)
    if input_path.is_dir():
        images = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
    else:
        images = [input_path]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = [(40, 220, 40), (0, 220, 255)]

    with torch.no_grad():
        for image_path in images:
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"skip unreadable: {image_path}")
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_tensor = predictor._transforms.forward_batch([image_rgb]).to(device)
            logits = model(image_tensor)
            logits = torch.nn.functional.interpolate(
                logits,
                size=image_bgr.shape[:2],
                mode="bilinear",
                align_corners=False,
            )[0]
            probs = logits.sigmoid().cpu().numpy()

            combined = image_bgr.copy()
            for idx, name in enumerate(class_names):
                mask = (probs[idx] > args.threshold).astype(np.uint8)
                np.save(output_dir / f"{image_path.stem}_{name}.npy", mask)
                cv2.imwrite(output_dir / f"{image_path.stem}_{name}.png", mask * 255)
                combined = overlay_mask(combined, mask, colors[idx % len(colors)])
            cv2.imwrite(output_dir / f"{image_path.stem}_overlay.jpg", combined)
            print(f"wrote {image_path.name}")


if __name__ == "__main__":
    main()
