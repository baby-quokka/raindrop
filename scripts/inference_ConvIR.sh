#!/usr/bin/env bash

# ConvIR NTIRE dev 추론 예시 스크립트

set -e

export CUDA_VISIBLE_DEVICES=0

MODEL_NAME="ConvIR_large"
CKPT_PATH="results/EXP_NAME/ckpt/Final.pkl"   # TODO: 실제 실험 이름으로 수정
OUTPUT_DIR="submission/ConvIR_large_EXP_NAME" # TODO: 실제 출력 폴더 이름으로 수정

python inference_ntire_dev.py \
  --model_name "${MODEL_NAME}" \
  --checkpoint "${CKPT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_multi_view_fusion \
  --use_self_ensemble \
  --fusion_alpha 0.2 \
  --scene_ranges_file data/valid/RaindropClarity_scene_ranges.txt

