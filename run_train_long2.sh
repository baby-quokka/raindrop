#!/bin/bash
#
# Stage1 long2: 200k iter, 8 실험 (GPU 번호 = tmux 창 번호)
#
# GPU 0: NAFNet-width48, 200k, 기존 loss
#   lambda_l1=1.0, lambda_lpips=0.1, lambda_msssim=0.1, lambda_fft=0.01, use_ema=False
# GPU 1: NAFNet-width64, 200k, 기존 loss
#
# GPU 2: ConvIR-base, 200k, l1ema
#   lambda_l1=1.5, lambda_lpips=0.1, lambda_msssim=0.1, lambda_fft=0.01, use_ema=True
# GPU 3: ConvIR-large, 200k, l1ema
#
# GPU 4: ConvIR-small, 200k,
#   L_total = 1.5·L1 + 0.1·L_MSSSIM + 0.05·L_FFT + 0.01·L_LPIPS
#   → lambda_l1=1.5, lambda_msssim=0.1, lambda_fft=0.05, lambda_lpips=0.01, use_ema=False
#
# GPU 5: NAFNet (width32), 200k,
#   L_total = 1.0·L1 + 0.2·L_MSSSIM + 0.1·L_FFT + 0·L_LPIPS
#   → lambda_l1=1.0, lambda_msssim=0.2, lambda_fft=0.1, lambda_lpips=0.0, use_ema=False
#
# GPU 6: ConvIR-base, 200k, l1ema + 낮은 LR(5e-5)
# GPU 7: NAFNet-width64, 200k, l1ema + 낮은 LR(5e-5)
#   (두 실험 모두 lambda_l1=1.5, lambda_lpips=0.1, lambda_msssim=0.1, lambda_fft=0.01, use_ema=True)
#
# 사용법:
#   bash run_train_long2.sh
#   bash run_train_long2.sh mysession
#

SESSION_NAME="${1:-train-long2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# conda 환경
CONDA_ENV="${CONDA_ENV:-raindrop}"
if [ -z "$CONDA_BASE" ] && command -v conda &>/dev/null; then
  CONDA_BASE="$(conda info --base 2>/dev/null)"
fi
if [ -z "$CONDA_BASE" ] && [ -d "/opt/conda" ]; then
  CONDA_BASE="/opt/conda"
fi
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
  PRE_CMD="source $CONDA_SH && conda activate $CONDA_ENV && "
else
  PRE_CMD=""
fi

# 200k iter, valid 10번 (20k마다)
NUM_ITER="${NUM_ITER:-200000}"
VALID_FREQ="${VALID_FREQ:-20000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-5000}"
CROP_SIZE="${CROP_SIZE:-256}"
NUM_WORKER="${NUM_WORKER:-4}"
PRINT_FREQ="${PRINT_FREQ:-100}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"

# 통신 포트 (기존 29500/29600과 겹치지 않게)
PORT_BASE="${PORT_BASE:-29700}"

# WandB
USE_WANDB="${USE_WANDB:-1}"
WANDB_FLAG=""
if [ "$USE_WANDB" = "1" ]; then
  WANDB_FLAG="--use_wandb"
fi

# Loss 계수 기본값
LAMBDA_LPIPS_ORIG="${LAMBDA_LPIPS_ORIG:-0.1}"
LAMBDA_LPIPS_05="${LAMBDA_LPIPS_05:-0.05}"   # 여기서는 GPU4에서 0.01로 따로 지정
LAMBDA_MSSSIM="${LAMBDA_MSSSIM:-0.1}"
LAMBDA_FFT="${LAMBDA_FFT:-0.01}"

# 공통 인자 (데이터/로깅 등)
COMMON_BASE="--train_data RaindropClarity --test_data RaindropClarity --num_worker $NUM_WORKER --print_freq $PRINT_FREQ --use_lpips $WANDB_FLAG"
COMMON_ITER="--num_iter $NUM_ITER --valid_freq $VALID_FREQ --lr_warmup_steps $LR_WARMUP_STEPS --crop $CROP_SIZE --batch_size $BATCH_SIZE --mixed_precision $MIXED_PRECISION"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session '$SESSION_NAME' already exists. Attach: tmux attach -t $SESSION_NAME"
  exit 0
fi

echo "tmux session '$SESSION_NAME' 생성 (200k iter, GPU 0~7)..."

########################
# GPU 0: NAFNet-width48, 기존 loss
########################
tmux new-session -d -s "$SESSION_NAME" -n "0-nafnet-w48-orig"
tmux send-keys -t "$SESSION_NAME:0" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+0)) main.py --model_name NAFNet_width48 --exp_suffix w48-orig200k --learning_rate 1e-4 $COMMON_BASE --lambda_l1 1.0 --lambda_lpips $LAMBDA_LPIPS_ORIG --lambda_msssim $LAMBDA_MSSSIM --lambda_fft $LAMBDA_FFT $COMMON_ITER" C-m

########################
# GPU 1: NAFNet-width64, 기존 loss
########################
tmux new-window -t "$SESSION_NAME" -n "1-nafnet-w64-orig"
tmux send-keys -t "$SESSION_NAME:1" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+1)) main.py --model_name NAFNet_width64 --exp_suffix w64-orig200k --learning_rate 1e-4 $COMMON_BASE --lambda_l1 1.0 --lambda_lpips $LAMBDA_LPIPS_ORIG --lambda_msssim $LAMBDA_MSSSIM --lambda_fft $LAMBDA_FFT $COMMON_ITER" C-m

########################
# GPU 2: ConvIR-base, l1ema
########################
tmux new-window -t "$SESSION_NAME" -n "2-convir-base-l1ema"
tmux send-keys -t "$SESSION_NAME:2" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+2)) main.py --model_name ConvIR_base --exp_suffix convir-base-l1ema200k --learning_rate 1e-4 $COMMON_BASE --lambda_l1 1.5 --lambda_lpips $LAMBDA_LPIPS_ORIG --lambda_msssim $LAMBDA_MSSSIM --lambda_fft $LAMBDA_FFT $COMMON_ITER --batch_size 2 --grad_accum 2" C-m

########################
# GPU 3,4: ConvIR-large, l1ema (2 GPU로 분산 → 12GB씩에서 OOM 방지, 유효 배치 2)
########################
tmux new-window -t "$SESSION_NAME" -n "3-convir-large-l1ema"
tmux send-keys -t "$SESSION_NAME:3" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=3,4 accelerate launch --num_processes 2 --main_process_port $((PORT_BASE+3)) main.py --model_name ConvIR_large --exp_suffix convir-large-l1ema200k --learning_rate 1e-4 $COMMON_BASE --lambda_l1 1.5 --lambda_lpips $LAMBDA_LPIPS_ORIG --lambda_msssim $LAMBDA_MSSSIM --lambda_fft $LAMBDA_FFT $COMMON_ITER --batch_size 1 --grad_accum 2 --use_ema" C-m

########################
# GPU 4: ConvIR-small, custom loss (1.5 L1 + 0.1 MSSSIM + 0.05 FFT + 0.01 LPIPS)
########################
tmux new-window -t "$SESSION_NAME" -n "4-convir-small-custom"
tmux send-keys -t "$SESSION_NAME:4" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=4 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+4)) main.py --model_name ConvIR --exp_suffix convir-small-custom200k --learning_rate 1e-4 $COMMON_BASE --lambda_l1 1.5 --lambda_lpips 0.01 --lambda_msssim 0.1 --lambda_fft 0.05 $COMMON_ITER" C-m

########################
# GPU 5: NAFNet (width32), custom loss (1.0 L1 + 0.2 MSSSIM + 0.1 FFT + 0·LPIPS)
########################
tmux new-window -t "$SESSION_NAME" -n "5-nafnet-w32-custom"
tmux send-keys -t "$SESSION_NAME:5" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=5 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+5)) main.py --model_name NAFNet --exp_suffix w32-custom200k --learning_rate 1e-4 $COMMON_BASE --lambda_l1 1.0 --lambda_lpips 0.0 --lambda_msssim 0.2 --lambda_fft 0.1 $COMMON_ITER" C-m

########################
# GPU 6: ConvIR (small), l1ema + 낮은 LR(5e-5)
########################
tmux new-window -t "$SESSION_NAME" -n "6-convir-base-l1ema-lr5e5"
tmux send-keys -t "$SESSION_NAME:6" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=6 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+6)) main.py --model_name ConvIR --exp_suffix convir-small-l1ema-lr5e5-200k --learning_rate 5e-5 $COMMON_BASE --lambda_l1 1.5 --lambda_lpips $LAMBDA_LPIPS_ORIG --lambda_msssim $LAMBDA_MSSSIM --lambda_fft $LAMBDA_FFT --use_ema $COMMON_ITER" C-m

########################
# GPU 7: NAFNet-width64, l1ema + 낮은 LR(5e-5)
########################
tmux new-window -t "$SESSION_NAME" -n "7-nafnet-w64-l1ema-lr5e5"
tmux send-keys -t "$SESSION_NAME:7" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=7 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+7)) main.py --model_name NAFNet_width64 --exp_suffix w64-l1ema-lr5e5-200k --learning_rate 5e-5 $COMMON_BASE --lambda_l1 1.5 --lambda_lpips $LAMBDA_LPIPS_ORIG --lambda_msssim $LAMBDA_MSSSIM --lambda_fft $LAMBDA_FFT --use_ema $COMMON_ITER" C-m

echo "=========================================="
echo "Tmux session: $SESSION_NAME (창 번호 = GPU 번호, 포트 $PORT_BASE~$((PORT_BASE+7)))"
echo "  0: NAFNet_width48  orig loss         GPU 0"
echo "  1: NAFNet_width64  orig loss         GPU 1"
echo "  2: ConvIR_base     l1ema             GPU 2"
echo "  3: ConvIR_large    l1ema             GPU 3"
echo "  4: ConvIR (small)  custom(1.5L1+0.1M+0.05F+0.01P) GPU 4"
echo "  5: NAFNet          custom(1.0L1+0.2M+0.1F+0P)      GPU 5"
echo "  6: ConvIR (small)  l1ema, lr=5e-5    GPU 6"
echo "  7: NAFNet_width64  l1ema, lr=5e-5    GPU 7"
echo "=========================================="
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Switch: Ctrl+b then 0~7"
echo "=========================================="

