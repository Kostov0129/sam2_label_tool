# SAM2 Label Tool

This repository contains a compact, reproducible workflow for building binary mask datasets with SAM2 and training a promptless SAM2 model for fixed object classes.

It is not limited to probes. You can label any object that needs a mask, then train a model with your own class names such as `plug`, `socket`, `button`, `probe_top`, or `probe_bump`.

It includes:

- Interactive SAM2-assisted mask labeling.
- Manual polygon/brush mask labeling.
- Dataset export with `images/`, `masks/`, `overlays/`, and `manifest.json`.
- Promptless SAM2 training with learned class tokens for fixed classes.
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
cd sam2_label_tool

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
raw/class_a/
  frame_0001.jpg
  frame_0002.jpg

raw/class_b/
  frame_0001.jpg
  frame_0002.jpg
```

Supported image suffixes: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`.

## 3. Label Masks with SAM2

Label `class_a`:

```bash
python wp/data_utils/probe_sam2_mask_labeler.py \
  --input_dir raw/class_a \
  --output_dir datasets/class_a \
  --class_name class_a \
  --model_size base
```

Label `class_b`:

```bash
python wp/data_utils/probe_sam2_mask_labeler.py \
  --input_dir raw/class_b \
  --output_dir datasets/class_b \
  --class_name class_b \
  --model_size base
```

Shortcut:

```bash
bash scripts/label_with_sam2.sh raw/class_a datasets/class_a class_a
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

Train a two-class model. The training script keeps historical argument names (`--whole_dir` and `--yellow_dir`), but the datasets can contain any two classes.

```bash
PYTHONUNBUFFERED=1 python wp/probe_train/train_probe_sam2.py \
  --whole_dir datasets/class_a \
  --yellow_dir datasets/class_b \
  --class_names class_a,class_b \
  --output_dir out/class_a_class_b_sam2 \
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

You can override class names used by the script by editing `scripts/train_promptless_sam2.sh` or running `train_probe_sam2.py` directly with `--class_names`.

For a smaller model:

```bash
MODEL_SIZE=tiny bash scripts/train_promptless_sam2.sh
```

Outputs:

```text
out/class_a_class_b_sam2/
  best.pt
  last.pt
  train_meta.json
```

## 6. Offline Inference

```bash
python wp/probe_train/infer_probe_sam2.py \
  --checkpoint out/class_a_class_b_sam2/best.pt \
  --input raw/test_images \
  --output_dir out/predictions \
  --threshold 0.5 \
  --device cuda
```

This writes per-class `.npy` masks, `.png` masks, and overlay previews.

Shortcut:

```bash
bash scripts/infer_image.sh out/class_a_class_b_sam2/best.pt raw/test_images out/predictions
```

## 7. RealSense Live Preview

```bash
python wp/probe_train/live_probe_sam2.py \
  --checkpoint out/class_a_class_b_sam2/best.pt \
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
bash scripts/live_realsense.sh out/class_a_class_b_sam2/best.pt
```

## 8. Manual Labeling Without SAM2

If SAM2 is unavailable or you want to manually fix masks:

```bash
python wp/data_utils/probe_mask_labeler.py \
  --input_dir raw/class_a \
  --output_dir datasets/class_a \
  --class_name class_a
```

## Notes

- SAM2 checkpoints are downloaded by `SAM2ImagePredictor.from_pretrained(...)` through Hugging Face. For offline use, run once online or pre-populate the Hugging Face cache.
- Use `HF_HUB_OFFLINE=1` when the checkpoint is already cached.
- This repository stores code only. Do not commit private image datasets or large `.pt` checkpoint files unless you intentionally publish them through Git LFS or GitHub Releases.
