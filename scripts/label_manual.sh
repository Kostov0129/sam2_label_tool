#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-examples/raw_images}"
OUTPUT_DIR="${2:-out/mask_dataset}"
CLASS_NAME="${3:-object}"

python wp/data_utils/probe_mask_labeler.py \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --class_name "${CLASS_NAME}"
