import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from probe_mask_labeler import ensure_dirs, list_images, load_mask, make_overlay, write_manifest, write_outputs, zip_dataset


MODEL_CHECKPOINTS = {
    "base": "facebook/sam2-hiera-base-plus",
    "small": "facebook/sam2-hiera-small",
    "tiny": "facebook/sam2-hiera-tiny",
}


class ProbeSAM2Annotator:
    def __init__(self, image_path, predictor, existing_mask=None):
        self.image_path = Path(image_path)
        self.image_bgr = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if self.image_bgr is None:
            raise ValueError(f"Failed to load image: {self.image_path}")

        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        if existing_mask is not None:
            self.mask = load_mask(existing_mask, self.image_bgr.shape)
        else:
            self.mask = np.zeros(self.image_bgr.shape[:2], dtype=np.uint8)

        self.predictor = predictor
        self.points = []
        self.labels = []
        self.manual_points = []
        self.history = [self.mask.copy()]
        self.mode = "sam"
        self.brush_radius = 12
        self.cursor = None
        self.dragging = False
        self.logits = None
        self.result = "save"
        self.is_predicting = False

    def set_image(self):
        self.predictor.set_image(self.image_rgb)

    def predict(self):
        import torch

        if not self.points:
            self.mask[:] = 0
            self.logits = None
            return

        point_coords = np.array(self.points, dtype=np.float32)
        point_labels = np.array(self.labels, dtype=np.int32)

        self.is_predicting = True
        try:
            if torch.cuda.is_available():
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, scores, logits = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True,
                    )
            else:
                with torch.inference_mode():
                    masks, scores, logits = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True,
                    )

            best_idx = int(np.argmax(scores))
            self.mask = np.where(masks[best_idx] > 0, 255, 0).astype(np.uint8)
            self.logits = logits[best_idx]
        finally:
            self.is_predicting = False

    def push_history(self):
        self.history.append(self.mask.copy())
        if len(self.history) > 30:
            self.history.pop(0)

    def cycle_mode(self):
        modes = ["sam", "polygon", "brush", "erase"]
        idx = modes.index(self.mode) if self.mode in modes else 0
        self.mode = modes[(idx + 1) % len(modes)]

    def set_mode(self, mode):
        if mode in {"sam", "polygon", "brush", "erase"}:
            self.mode = mode

    def fill_polygon(self):
        if len(self.manual_points) < 3:
            return
        self.push_history()
        pts = np.array(self.manual_points, dtype=np.int32)
        cv2.fillPoly(self.mask, [pts], 255)
        self.manual_points = []

    def draw_brush(self, x, y):
        self.push_history()
        value = 0 if self.mode == "erase" else 255
        cv2.circle(self.mask, (x, y), self.brush_radius, value, -1)

    def on_mouse(self, event, x, y, flags, _param):
        self.cursor = (x, y)
        if self.mode == "sam":
            if event == cv2.EVENT_LBUTTONDOWN:
                label = 0 if flags & cv2.EVENT_FLAG_CTRLKEY else 1
                self.points.append((x, y))
                self.labels.append(label)
                self.predict()
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.points.append((x, y))
                self.labels.append(0)
                self.predict()
            return

        if self.mode == "polygon":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.manual_points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.fill_polygon()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.draw_brush(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            value = 0 if self.mode == "erase" else 255
            cv2.circle(self.mask, (x, y), self.brush_radius, value, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def add_sam_point(self, x, y, flags):
        if self.mode == "sam":
            label = 0 if flags & cv2.EVENT_FLAG_CTRLKEY else 1
            self.points.append((x, y))
            self.labels.append(label)
            self.predict()

    def undo(self):
        if self.mode == "polygon" and self.manual_points:
            self.manual_points.pop()
            return
        if self.mode == "sam" and self.points:
            self.points.pop()
            self.labels.pop()
            self.predict()
            return
        if len(self.history) > 1:
            self.history.pop()
            self.mask = self.history[-1].copy()

    def clear(self):
        self.points = []
        self.labels = []
        self.manual_points = []
        self.mask[:] = 0
        self.logits = None

    def draw_points(self, view):
        for (x, y), label in zip(self.points, self.labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(view, (x, y), 6, color, -1)
            cv2.circle(view, (x, y), 7, (255, 255, 255), 1)

    def draw_manual_points(self, view):
        if not self.manual_points:
            return
        pts = np.array(self.manual_points, dtype=np.int32)
        for point in pts:
            cv2.circle(view, tuple(point), 4, (0, 255, 255), -1)
        if len(pts) > 1:
            cv2.polylines(view, [pts], False, (0, 255, 255), 2)

    def mode_label(self):
        labels = {
            "sam": "SAM2 / 1: left include, right or Ctrl+left exclude",
            "polygon": "POLYGON / 2: click outline, right-click or F fill",
            "brush": "BRUSH / 3: drag to add area",
            "erase": "ERASE / 4: drag to remove area",
        }
        return labels.get(self.mode, self.mode)

    def help_lines(self):
        status = "predicting..." if self.is_predicting else f"{len(self.points)} points"
        pos_count = sum(1 for label in self.labels if label == 1)
        neg_count = sum(1 for label in self.labels if label == 0)
        return [
            f"Image: {self.image_path.name}",
            f"Mode: {self.mode_label()} | TAB cycles | brush={self.brush_radius}px",
            f"SAM status: {status} | positive green={pos_count} | negative red={neg_count}",
            "U undo | C clear | +/- brush size | SPACE/S save next | K skip | Q/ESC quit",
        ]

    def show(self):
        window = "Probe SAM2 mask labeler"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1100, 800)
        cv2.setMouseCallback(window, self.on_mouse)

        while True:
            view = make_overlay(self.image_bgr, self.mask)
            self.draw_points(view)
            self.draw_manual_points(view)
            if self.cursor is not None and self.mode in {"brush", "erase"}:
                cv2.circle(view, self.cursor, self.brush_radius, (255, 255, 255), 1)

            panel_h = 146
            panel = view[:panel_h, :].copy()
            dark = np.zeros_like(panel)
            view[:panel_h, :] = cv2.addWeighted(panel, 0.25, dark, 0.75, 0)

            y = 24
            for line in self.help_lines():
                cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 1, cv2.LINE_AA)
                y += 26

            cv2.imshow(window, view)
            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue
            if key in (ord("q"), 27):
                self.result = "quit"
                break
            if key in (ord("s"), ord(" ")):
                self.fill_polygon()
                self.result = "save"
                break
            if key == ord("k"):
                self.result = "skip"
                break
            if key == 9:
                self.cycle_mode()
            elif key == ord("1"):
                self.set_mode("sam")
            elif key in (ord("2"), ord("p")):
                self.set_mode("polygon")
            elif key in (ord("3"), ord("b")):
                self.set_mode("brush")
            elif key in (ord("4"), ord("e")):
                self.set_mode("erase")
            elif key == ord("f") or key == 13:
                self.fill_polygon()
            elif key == ord("u"):
                self.undo()
            elif key == ord("c"):
                self.clear()
            elif key in (ord("+"), ord("=")):
                self.brush_radius = min(120, self.brush_radius + 2)
            elif key in (ord("-"), ord("_")):
                self.brush_radius = max(1, self.brush_radius - 2)

        cv2.destroyWindow(window)
        return self.result, self.mask


def main():
    parser = argparse.ArgumentParser(description="Annotate probe surface masks with SAM2 point prompts.")
    parser.add_argument("--input_dir", required=True, help="Directory containing probe images.")
    parser.add_argument("--output_dir", default="out/probe_surface_dataset", help="Directory for images, masks, overlays, manifest.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of images to label.")
    parser.add_argument("--overwrite", action="store_true", help="Relabel images whose masks already exist.")
    parser.add_argument("--model_size", default="base", choices=sorted(MODEL_CHECKPOINTS), help="SAM2 model size.")
    parser.add_argument("--class_name", default="probe_surface", help="Foreground class name stored in manifest.json.")
    parser.add_argument("--zip", dest="zip_path", default=None, help="Optional path to write a zip package.")
    args = parser.parse_args()

    output_dir = ensure_dirs(args.output_dir)
    images = list_images(args.input_dir)
    if args.limit:
        images = images[: args.limit]

    records = []
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            records = json.load(f).get("items", [])

    checkpoint = MODEL_CHECKPOINTS[args.model_size]
    print(f"Loading SAM2 model: {checkpoint}")
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise SystemExit(
            "SAM2 is not available in this Python environment. Activate the conda environment "
            "where torch and sam2 are installed, then rerun this command."
        ) from exc

    predictor = SAM2ImagePredictor.from_pretrained(checkpoint)

    print(f"Found {len(images)} images.")
    for idx, image_path in enumerate(images, start=1):
        mask_path = output_dir / "masks" / f"{image_path.stem}.npy"
        if mask_path.exists() and not args.overwrite:
            print(f"[{idx}/{len(images)}] skip existing: {image_path.name}")
            continue

        print(f"[{idx}/{len(images)}] labeling: {image_path.name}")
        annotator = ProbeSAM2Annotator(image_path, predictor, existing_mask=mask_path)
        annotator.set_image()
        result, mask = annotator.show()

        if result == "quit":
            print("Stopped by user.")
            break
        if result == "skip":
            print(f"Skipped: {image_path.name}")
            continue

        record = write_outputs(image_path, annotator.image_bgr, mask, output_dir, class_name=args.class_name)
        records = [r for r in records if Path(r["image"]).stem != image_path.stem]
        records.append(record)
        write_manifest(records, output_dir, class_name=args.class_name)
        print(f"Saved mask: {record['mask']}")

    write_manifest(records, output_dir, class_name=args.class_name)
    if args.zip_path:
        zip_dataset(output_dir, args.zip_path)
        print(f"Wrote zip: {args.zip_path}")


if __name__ == "__main__":
    main()
