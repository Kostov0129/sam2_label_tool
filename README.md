# SAM2 Label Tool

Minimal SAM2-assisted binary mask labeler.

## Install

```bash
git clone https://github.com/Kostov0129/sam2_label_tool.git
cd sam2_label_tool

bash scripts/install_env.sh
conda activate sam2_label_tool
```

If you already have a SAM2 environment:

```bash
pip install -r requirements.txt
pip install -e /path/to/sam2
```

## Prepare Images

```text
raw/my_object/
  frame_0001.jpg
  frame_0002.jpg
```

Supported: `.jpg .jpeg .png .bmp .webp .tif .tiff`

## Run

```bash
bash scripts/label.sh raw/my_object datasets/my_object my_object
```

Arguments:

```text
input_dir output_dir class_name
```

Optional:

```bash
MODEL_SIZE=tiny bash scripts/label.sh raw/my_object datasets/my_object my_object
OVERWRITE=1 bash scripts/label.sh raw/my_object datasets/my_object my_object
ZIP_PATH=out/my_object.zip bash scripts/label.sh raw/my_object datasets/my_object my_object
```

## Controls

- `1`: SAM2 points
- Left click: positive point
- Right click / `Ctrl + Left click`: negative point
- `2` / `p`: polygon
- `3` / `b`: brush add
- `4` / `e`: erase
- Right click / `f` / Enter: fill polygon
- `u`: undo
- `c`: clear
- `+` / `-`: brush size
- Space / `s`: save next
- `k`: skip
- `q` / Esc: quit

## Output

```text
datasets/my_object/
  images/
  masks/
  overlays/
  manifest.json
```

Masks are `.npy` arrays:

```text
0 = background
1 = object
```

## Python Entry

```bash
python -m sam2_label_tool.labeler \
  --input_dir raw/my_object \
  --output_dir datasets/my_object \
  --class_name my_object \
  --model_size base
```

Options:

```text
--model_size base|small|tiny
--limit N
--overwrite
--brush_radius N
--zip out/dataset.zip
```

For offline use, cache SAM2 once, then run with `HF_HUB_OFFLINE=1`.
