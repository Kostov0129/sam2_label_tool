#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:-examples/raw_images}"
OUTPUT_DIR="${2:-out/mask_dataset}"
CLASS_NAME="${3:-object}"
MODEL_SIZE="${MODEL_SIZE:-base}"
OVERWRITE="${OVERWRITE:-0}"
ZIP_PATH="${ZIP_PATH:-}"

args=(
  python -m sam2_label_tool.labeler
  --input_dir "${INPUT_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --class_name "${CLASS_NAME}"
  --model_size "${MODEL_SIZE}"
)

if [ "${OVERWRITE}" = "1" ]; then
  args+=(--overwrite)
fi

if [ -n "${ZIP_PATH}" ]; then
  args+=(--zip "${ZIP_PATH}")
fi

"${args[@]}"
