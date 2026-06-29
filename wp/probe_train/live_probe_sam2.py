import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wp.probe_train.train_probe_sam2 import MODEL_IDS, PromptlessProbeSAM2


def overlay_mask(image_bgr, mask, color, alpha=0.45):
    out = image_bgr.copy()
    color_layer = np.zeros_like(out)
    color_layer[:, :] = color
    fg = mask > 0
    if np.any(fg):
        out[fg] = cv2.addWeighted(out[fg], 1.0 - alpha, color_layer[fg], alpha, 0)
    return out


def clean_mask(mask, close_kernel=7, min_area=600, largest_only=True):
    mask = (mask > 0).astype(np.uint8)
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    h, w = mask.shape[:2]
    flood = (mask * 255).copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    mask = np.where((mask > 0) | (holes > 0), 1, 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros_like(mask)
    if largest_only:
        best = 1 + int(np.argmax(areas))
        if stats[best, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == best] = 1
        return keep

    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == label_id] = 1
    return keep


def draw_text(image, lines):
    y = 26
    for line in lines:
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28


def open_realsense(width, height, fps):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    return pipeline, profile


def main():
    parser = argparse.ArgumentParser(description="Live RealSense probe segmentation preview.")
    parser.add_argument("--checkpoint", default="out/probe_sam2_train/best.pt")
    parser.add_argument("--output_dir", default="out/probe_live_captures")
    parser.add_argument("--model_size", default=None, choices=[None, *sorted(MODEL_IDS)])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--infer_every", type=int, default=3, help="Run model every N frames.")
    parser.add_argument("--class_id", type=int, default=-1, help="-1 runs all classes; 0=first class; 1=second class.")
    parser.add_argument("--close_kernel", type=int, default=9, help="Morphological cleanup kernel size. Use 0 to disable.")
    parser.add_argument("--min_area", type=int, default=800, help="Remove connected components below this pixel area.")
    parser.add_argument("--largest_only", action="store_true", default=True, help="Keep only the largest connected component per class.")
    parser.add_argument("--raw_mask", action="store_true", help="Disable mask cleanup.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from sam2.sam2_image_predictor import SAM2ImagePredictor

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_size = args.model_size or ckpt.get("args", {}).get("model_size", "base")
    all_class_names = ckpt.get("class_names", ["probe_whole", "probe_yellow_side"])
    if args.class_id >= 0:
        if args.class_id >= len(all_class_names):
            raise SystemExit(f"--class_id {args.class_id} is out of range for classes: {all_class_names}")
        selected_indices = [args.class_id]
        class_names = [all_class_names[args.class_id]]
    else:
        selected_indices = None
        class_names = all_class_names
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {model_size} on {device}")
    predictor = SAM2ImagePredictor.from_pretrained(MODEL_IDS[model_size], device=str(device))
    model = PromptlessProbeSAM2(predictor.model, num_classes=len(all_class_names)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    print(f"Opening RealSense color stream: {args.width}x{args.height}@{args.fps}")
    pipeline, _profile = open_realsense(args.width, args.height, args.fps)

    if len(class_names) == 1:
        window = f"Probe live SAM2: red={class_names[0]}"
    else:
        window = "Probe live SAM2: red=class0 blue=class1"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1100, 800)

    frame_idx = 0
    last_masks = None
    last_infer_ms = 0.0
    last_time = time.time()
    fps_smooth = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame_bgr = np.asanyarray(color_frame.get_data())
            display = frame_bgr.copy()

            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt) if fps_smooth else 1.0 / dt

            if frame_idx % max(1, args.infer_every) == 0:
                image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image_tensor = predictor._transforms.forward_batch([image_rgb]).to(device)
                start = time.time()
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                    logits = model(image_tensor, class_indices=selected_indices)
                    logits = torch.nn.functional.interpolate(
                        logits,
                        size=frame_bgr.shape[:2],
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                    probs = logits.sigmoid().float().cpu().numpy()
                raw_masks = [(probs[i] > args.threshold).astype(np.uint8) for i in range(len(class_names))]
                if args.raw_mask:
                    last_masks = raw_masks
                else:
                    kernel = max(0, args.close_kernel)
                    if kernel > 0 and kernel % 2 == 0:
                        kernel += 1
                    last_masks = [
                        clean_mask(mask, close_kernel=kernel, min_area=args.min_area, largest_only=args.largest_only)
                        for mask in raw_masks
                    ]
                last_infer_ms = (time.time() - start) * 1000.0

            areas = []
            if last_masks is not None:
                colors = [(0, 0, 255)] if len(class_names) == 1 else [(0, 0, 255), (255, 0, 0)]
                for i, mask in enumerate(last_masks):
                    display = overlay_mask(display, mask, colors[i % len(colors)])
                    areas.append(int(mask.sum()))

            draw_text(
                display,
                [
                    "q/ESC quit | s save frame+mask",
                    f"fps={fps_smooth:.1f} infer={last_infer_ms:.0f}ms every={args.infer_every} threshold={args.threshold}",
                    " | ".join(f"{name} area={areas[i] if i < len(areas) else 0}" for i, name in enumerate(class_names)),
                ],
            )

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(output_dir / f"{stamp}_overlay.jpg"), display)
                cv2.imwrite(str(output_dir / f"{stamp}_frame.jpg"), frame_bgr)
                if last_masks is not None:
                    for name, mask in zip(class_names, last_masks):
                        np.save(output_dir / f"{stamp}_{name}.npy", mask)
                        cv2.imwrite(str(output_dir / f"{stamp}_{name}.png"), mask * 255)
                print(f"saved {stamp}")
            frame_idx += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
