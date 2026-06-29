#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-checkpoints/best.pt}"

python wp/probe_train/live_probe_sam2.py \
  --checkpoint "${CHECKPOINT}" \
  --class_id "${CLASS_ID:--1}" \
  --threshold "${THRESHOLD:-0.5}" \
  --width "${WIDTH:-640}" \
  --height "${HEIGHT:-480}" \
  --fps "${FPS:-30}" \
  --infer_every "${INFER_EVERY:-3}" \
  --device "${DEVICE:-cuda}"
