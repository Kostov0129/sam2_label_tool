#!/usr/bin/env bash
set -euo pipefail

TOP_DIR="${TOP_DIR:-${CLASS_A_DIR:-out/class_a_dataset}}"
BUMP_DIR="${BUMP_DIR:-${CLASS_B_DIR:-out/class_b_dataset}}"
OUTPUT_DIR="${OUTPUT_DIR:-out/class_a_class_b_sam2}"
MODEL_SIZE="${MODEL_SIZE:-base}"
CLASS_NAMES="${CLASS_NAMES:-class_a,class_b}"

PYTHONUNBUFFERED=1 python wp/probe_train/train_probe_sam2.py \
  --whole_dir "${TOP_DIR}" \
  --yellow_dir "${BUMP_DIR}" \
  --class_names "${CLASS_NAMES}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_size "${MODEL_SIZE}" \
  --epochs "${EPOCHS:-120}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --lr "${LR:-1e-5}" \
  --weight_decay "${WEIGHT_DECAY:-1e-4}" \
  --val_split "${VAL_SPLIT:-0.15}" \
  --auto_stop \
  --min_epochs "${MIN_EPOCHS:-30}" \
  --early_stop_patience "${EARLY_STOP_PATIENCE:-20}" \
  --early_stop_min_delta "${EARLY_STOP_MIN_DELTA:-0.0015}" \
  --device "${DEVICE:-cuda}" \
  --num_workers "${NUM_WORKERS:-2}"
