#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-checkpoints/best.pt}"
INPUT="${2:-examples/raw_images}"
OUTPUT_DIR="${3:-out/predictions}"

python wp/probe_train/infer_probe_sam2.py \
  --checkpoint "${CHECKPOINT}" \
  --input "${INPUT}" \
  --output_dir "${OUTPUT_DIR}" \
  --threshold "${THRESHOLD:-0.5}" \
  --device "${DEVICE:-cuda}"
