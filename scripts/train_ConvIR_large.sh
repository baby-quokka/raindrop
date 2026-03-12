#!/usr/bin/env bash

# ConvIR-large 학습 예시 스크립트 예시

set -e

export CUDA_VISIBLE_DEVICES=0,1

accelerate launch \
  --num_processes 2 \
  --main_process_port 29711 \
  --ddp_timeout 3600 \
  main.py \
  --model_name ConvIR_large \
  --exp_suffix batch8-warmup70k \
  --learning_rate 3e-4 \
  --train_data RaindropClarity \
  --test_data RaindropClarity \
  --num_worker 4 \
  --print_freq 100 \
  --use_lpips \
  --lambda_l1 1.5 \
  --lambda_lpips 0.1 \
  --lambda_msssim 0.1 \
  --lambda_fft 0.01 \
  --use_ema \
  --num_iter 200000 \
  --valid_freq 20000 \
  --lr_warmup_steps 70000 \
  --crop 256 \
  --batch_size 4 \
  --mixed_precision no \
  --lr_scheduler cosine

