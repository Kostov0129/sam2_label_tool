# Probe SAM2 Labeling and Promptless Mask Training

This repository contains a compact, reproducible workflow for building probe mask datasets with SAM2 and training a promptless SAM2 model for fixed classes such as `probe_top` and `probe_bump`.

It includes:

- Interactive SAM2-assisted mask labeling.
- Manual polygon/brush mask labeling.
- Dataset export with `images/`, `masks/`, `overlays/`, and `manifest.json`.
- Promptless SAM2 training with learned class tokens.
- Offline image/directory inference.
- RealSense live preview.

Large files are intentionally not tracked:

- Raw image datasets.
- Trained checkpoints.
- Generated outputs.

Place those under `datasets/`, `raw/`, `out/`, or `checkpoints/` locally.

## 1. Install

```bash
git clone <this-repo-url>
cd sam2-probe-labeling

bash scripts/install_env.sh
conda activate probe_sam2
```

The installer creates a conda environment and installs the official Facebook SAM2 repository into `third_party/sam2`.

If you already have a SAM2 environment, install only the Python requirements:

```bash
pip install -r requirements.txt
pip install -e /path/to/sam2
```

## 2. Prepare Raw Images

Put images in one folder:

```text
raw/probe_top/
  frame_0001.jpg
  frame_0002.jpg

raw/probe_bump/
  frame_0001.jpg
  frame_0002.jpg
```

Supported image suffixes: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`.

## 3. Label Masks with SAM2

Label `probe_top`:

```bash
python wp/data_utils/probe_sam2_mask_labeler.py \
  --input_dir raw/probe_top \
  --output_dir datasets/probe_top \
  --class_name probe_top \
  --model_size base
```

Label `probe_bump`:

```bash
python wp/data_utils/probe_sam2_mask_labeler.py \
  --input_dir raw/probe_bump \
  --output_dir datasets/probe_bump \
  --class_name probe_bump \
  --model_size base
```

Shortcut:

```bash
bash scripts/label_with_sam2.sh raw/probe_top datasets/probe_top probe_top
```

### Labeler Controls

- `1`: SAM2 prompt mode.
- Left click: positive SAM2 point.
- Right click or `Ctrl + Left click`: negative SAM2 point.
- `2` / `p`: polygon mode.
- `3` / `b`: brush add mode.
- `4` / `e`: erase mode.
- Right click / `f` / Enter: fill polygon.
- `u`: undo.
- `c`: clear.
- `+` / `-`: brush size.
- Space / `s`: save and continue.
- `k`: skip image.
- `q` / Esc: quit.

## 4. Dataset Format

Each labeled dataset has:

```text
datasets/probe_top/
  images/
  masks/
  overlays/
  manifest.json
```

Masks are `.npy` binary arrays:

- `0`: background
- `1`: foreground class

## 5. Train Promptless SAM2

Train a two-class `probe_top` / `probe_bump` model:

```bash
PYTHONUNBUFFERED=1 python wp/probe_train/train_probe_sam2.py \
  --whole_dir datasets/probe_top \
  --yellow_dir datasets/probe_bump \
  --class_names probe_top,probe_bump \
  --output_dir out/probe_top_bump_sam2 \
  --model_size base \
  --epochs 120 \
  --batch_size 1 \
  --lr 1e-5 \
  --weight_decay 1e-4 \
  --val_split 0.15 \
  --auto_stop \
  --min_epochs 30 \
  --early_stop_patience 20 \
  --early_stop_min_delta 0.0015 \
  --device cuda \
  --num_workers 2
```

Shortcut:

```bash
TOP_DIR=datasets/probe_top \
BUMP_DIR=datasets/probe_bump \
OUTPUT_DIR=out/probe_top_bump_sam2 \
MODEL_SIZE=base \
bash scripts/train_promptless_sam2.sh
```

For a smaller model:

```bash
MODEL_SIZE=tiny bash scripts/train_promptless_sam2.sh
```

Outputs:

```text
out/probe_top_bump_sam2/
  best.pt
  last.pt
  train_meta.json
```

## 6. Offline Inference

```bash
python wp/probe_train/infer_probe_sam2.py \
  --checkpoint out/probe_top_bump_sam2/best.pt \
  --input raw/test_images \
  --output_dir out/predictions \
  --threshold 0.5 \
  --device cuda
```

This writes per-class `.npy` masks, `.png` masks, and overlay previews.

Shortcut:

```bash
bash scripts/infer_image.sh out/probe_top_bump_sam2/best.pt raw/test_images out/predictions
```

## 7. RealSense Live Preview

```bash
python wp/probe_train/live_probe_sam2.py \
  --checkpoint out/probe_top_bump_sam2/best.pt \
  --class_id -1 \
  --threshold 0.5 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --infer_every 3 \
  --device cuda
```

`--class_id -1` runs all classes. Use `0` for the first class and `1` for the second class.

Shortcut:

```bash
bash scripts/live_realsense.sh out/probe_top_bump_sam2/best.pt
```

## 8. Manual Labeling Without SAM2

If SAM2 is unavailable or you want to manually fix masks:

```bash
python wp/data_utils/probe_mask_labeler.py \
  --input_dir raw/probe_top \
  --output_dir datasets/probe_top \
  --class_name probe_top
```

## Notes

- SAM2 checkpoints are downloaded by `SAM2ImagePredictor.from_pretrained(...)` through Hugging Face. For offline use, run once online or pre-populate the Hugging Face cache.
- Use `HF_HUB_OFFLINE=1` when the checkpoint is already cached.
- This repository stores code only. Do not commit private image datasets or large `.pt` checkpoint files unless you intentionally publish them through Git LFS or GitHub Releases.
