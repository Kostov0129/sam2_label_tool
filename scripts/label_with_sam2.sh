#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-examples/raw_images}"
OUTPUT_DIR="${2:-out/probe_surface_dataset}"
CLASS_NAME="${3:-probe_surface}"
MODEL_SIZE="${MODEL_SIZE:-base}"

python wp/data_utils/probe_sam2_mask_labeler.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --class_name "${CLASS_NAME}" \
  --model_size "${MODEL_SIZE}"
