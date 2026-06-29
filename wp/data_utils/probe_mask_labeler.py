import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(input_dir):
    paths = []
    for path in Path(input_dir).iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            paths.append(path)
    return sorted(paths)


def ensure_dirs(output_dir):
    output_dir = Path(output_dir)
    for name in ["images", "masks", "overlays"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    return output_dir


def load_mask(mask_path, shape):
    mask_path = Path(mask_path)
    h, w = shape[:2]
    if not mask_path.exists():
        return np.zeros((h, w), dtype=np.uint8)
    if mask_path.suffix.lower() == ".npy":
        mask = np.load(mask_path)
    else:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape[:2] != (h, w):
        return np.zeros((h, w), dtype=np.uint8)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def copy_image(src_path, dst_path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def make_overlay(image_bgr, mask, alpha=0.45, points=None, brush_radius=None, cursor=None):
    overlay = image_bgr.copy()
    color_layer = np.zeros_like(image_bgr)
    color_layer[:, :, 1] = 220
    color_layer[:, :, 2] = 40
    foreground = mask > 0
    if np.any(foreground):
        overlay[foreground] = cv2.addWeighted(
            image_bgr[foreground], 1.0 - alpha, color_layer[foreground], alpha, 0
        )

    if points:
        pts = np.array(points, dtype=np.int32)
        for point in pts:
            cv2.circle(overlay, tuple(point), 4, (0, 255, 255), -1)
        if len(pts) > 1:
            cv2.polylines(overlay, [pts], False, (0, 255, 255), 2)

    if cursor is not None and brush_radius is not None:
        cv2.circle(overlay, cursor, brush_radius, (255, 255, 255), 1)

    return overlay


class ProbeMaskAnnotator:
    def __init__(self, image_path, existing_mask=None, brush_radius=12):
        self.image_path = Path(image_path)
        self.image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if self.image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")

        if existing_mask is not None:
            self.mask = load_mask(existing_mask, self.image.shape)
        else:
            self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

        self.points = []
        self.history = [self.mask.copy()]
        self.mode = "polygon"
        self.brush_radius = brush_radius
        self.cursor = None
        self.dragging = False
        self.result = "save"

    def set_mode(self, mode):
        if mode in {"polygon", "brush", "erase"}:
            self.mode = mode

    def cycle_mode(self):
        modes = ["polygon", "brush", "erase"]
        idx = modes.index(self.mode) if self.mode in modes else 0
        self.mode = modes[(idx + 1) % len(modes)]

    def mode_label(self):
        labels = {
            "polygon": "POLYGON / 1 / P: click outline, right-click or F to fill",
            "brush": "BRUSH / 2 / B: paint included area",
            "erase": "ERASE / 3 / E: remove extra area",
        }
        return labels.get(self.mode, self.mode)

    def push_history(self):
        self.history.append(self.mask.copy())
        if len(self.history) > 30:
            self.history.pop(0)

    def undo(self):
        if self.points:
            self.points.pop()
            return
        if len(self.history) > 1:
            self.history.pop()
            self.mask = self.history[-1].copy()

    def fill_polygon(self):
        if len(self.points) < 3:
            return
        self.push_history()
        pts = np.array(self.points, dtype=np.int32)
        cv2.fillPoly(self.mask, [pts], 255)
        self.points = []

    def clear_mask(self):
        self.push_history()
        self.mask[:] = 0
        self.points = []

    def draw_brush(self, x, y):
        self.push_history()
        value = 0 if self.mode == "erase" else 255
        cv2.circle(self.mask, (x, y), self.brush_radius, value, -1)

    def on_mouse(self, event, x, y, flags, _param):
        self.cursor = (x, y)
        if self.mode == "polygon":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
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

    def help_lines(self):
        return [
            f"Image: {self.image_path.name}",
            f"Mode: {self.mode_label()} | brush={self.brush_radius}px | TAB cycles modes",
            "Mouse: left click/drag works in current mode | right click/F fills polygon",
            "U undo | C clear | +/- brush size | SPACE/S save next | K skip | Q/ESC quit",
        ]

    def show(self):
        window = "Probe surface mask labeler"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1100, 800)
        cv2.setMouseCallback(window, self.on_mouse)

        while True:
            view = make_overlay(
                self.image,
                self.mask,
                points=self.points,
                brush_radius=self.brush_radius if self.mode in {"brush", "erase"} else None,
                cursor=self.cursor,
            )
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

            if key in (255,):
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
            elif key in (ord("p"), ord("1")):
                self.set_mode("polygon")
            elif key in (ord("b"), ord("2")):
                self.set_mode("brush")
            elif key in (ord("e"), ord("3")):
                self.set_mode("erase")
            elif key == ord("f") or key == 13:
                self.fill_polygon()
            elif key == ord("u"):
                self.undo()
            elif key == ord("c"):
                self.clear_mask()
            elif key in (ord("+"), ord("=")):
                self.brush_radius = min(120, self.brush_radius + 2)
            elif key in (ord("-"), ord("_")):
                self.brush_radius = max(1, self.brush_radius - 2)

        cv2.destroyWindow(window)
        return self.result, self.mask


def write_outputs(image_path, image_bgr, mask, output_dir, class_name="probe_surface"):
    output_dir = Path(output_dir)
    stem = image_path.stem
    image_dst = output_dir / "images" / f"{stem}.jpg"
    mask_dst = output_dir / "masks" / f"{stem}.npy"
    overlay_dst = output_dir / "overlays" / f"{stem}.jpg"

    cv2.imwrite(str(image_dst), image_bgr)
    binary_mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    np.save(mask_dst, binary_mask)
    cv2.imwrite(str(overlay_dst), make_overlay(image_bgr, mask))

    return {
        "image": str(image_dst.relative_to(output_dir)),
        "mask": str(mask_dst.relative_to(output_dir)),
        "overlay": str(overlay_dst.relative_to(output_dir)),
        "class_name": class_name,
        "mask_values": {"background": 0, class_name: 1},
    }


def write_manifest(records, output_dir, class_name="probe_surface"):
    manifest = {
        "classes": [{"id": 1, "name": class_name, "mask_value": 255}],
        "items": records,
    }
    with open(Path(output_dir) / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def zip_dataset(output_dir, zip_path):
    output_dir = Path(output_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.resolve() != zip_path.resolve():
                zf.write(path, path.relative_to(output_dir))


def main():
    parser = argparse.ArgumentParser(description="Annotate probe surface binary masks.")
    parser.add_argument("--input_dir", required=True, help="Directory containing probe images.")
    parser.add_argument("--output_dir", default="out/probe_surface_dataset", help="Directory for images, masks, overlays, manifest.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of images to label.")
    parser.add_argument("--overwrite", action="store_true", help="Relabel images whose masks already exist.")
    parser.add_argument("--brush_radius", type=int, default=12, help="Initial brush radius in pixels.")
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

    print(f"Found {len(images)} images.")
    for idx, image_path in enumerate(images, start=1):
        mask_path = output_dir / "masks" / f"{image_path.stem}.npy"
        if mask_path.exists() and not args.overwrite:
            print(f"[{idx}/{len(images)}] skip existing: {image_path.name}")
            continue

        print(f"[{idx}/{len(images)}] labeling: {image_path.name}")
        annotator = ProbeMaskAnnotator(image_path, existing_mask=mask_path, brush_radius=args.brush_radius)
        result, mask = annotator.show()

        if result == "quit":
            print("Stopped by user.")
            break
        if result == "skip":
            print(f"Skipped: {image_path.name}")
            continue

        record = write_outputs(image_path, annotator.image, mask, output_dir, class_name=args.class_name)
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
