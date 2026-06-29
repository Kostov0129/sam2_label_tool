import json
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


def write_outputs(image_path, image_bgr, mask, output_dir, class_name="object"):
    output_dir = Path(output_dir)
    stem = Path(image_path).stem
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


def write_manifest(records, output_dir, class_name="object"):
    manifest = {
        "classes": [{"id": 1, "name": class_name, "mask_value": 1}],
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


def copy_tree_like_dataset(src_dir, dst_dir):
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in Path(src_dir).iterdir():
        if path.is_file():
            shutil.copy2(path, dst_dir / path.name)

