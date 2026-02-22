#!/bin/bash
#
# ConvIR-base LR schedule 실험용 임시 스크립트 (250k iter)
# - 공통: ConvIR_base, l1ema (lambda_l1=1.5, lambda_lpips=0.1, lambda_msssim=0.1, lambda_fft=0.01, EMA 사용)
# - 유효 배치: 4 (batch_size=2, grad_accum=2, GPU 1장당)
#
# GPU 0: 실험 B - LR 1.5e-4, constant_with_warmup
# GPU 1: 실험 C - LR 1.5e-4, cosine
# GPU 2: 실험 D - LR 1.5e-4, polynomial (linear decay)
# GPU 5: 실험 E - LR 1.5e-4, step_custom (3-step decay)
#
# 사용법:
#   bash run_train_lr_schedule_long_temp.sh
#   bash run_train_lr_schedule_long_temp.sh mysession
#

SESSION_NAME="${1:-train-lr-sched-temp}"
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

# 250k iter, valid 10번 (25k마다)
NUM_ITER="${NUM_ITER:-250000}"
VALID_FREQ="${VALID_FREQ:-25000}"
BATCH_SIZE="${BATCH_SIZE:-4}"        # 기본값 (ConvIR-base는 아래에서 2로 override)
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-5000}"
CROP_SIZE="${CROP_SIZE:-256}"
NUM_WORKER="${NUM_WORKER:-4}"
PRINT_FREQ="${PRINT_FREQ:-100}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"

# 통신 포트 (다른 스크립트와 겹치지 않게)
PORT_BASE="${PORT_BASE:-29900}"

# WandB
USE_WANDB="${USE_WANDB:-1}"
WANDB_FLAG=""
if [ "$USE_WANDB" = "1" ]; then
  WANDB_FLAG="--use_wandb"
fi

# Loss 계수 (l1ema 설정)
LAMBDA_L1="${LAMBDA_L1:-1.5}"
LAMBDA_LPIPS="${LAMBDA_LPIPS:-0.1}"
LAMBDA_MSSSIM="${LAMBDA_MSSSIM:-0.1}"
LAMBDA_FFT="${LAMBDA_FFT:-0.01}"

# 공통 인자 (데이터/로깅 등)
COMMON_BASE="--train_data RaindropClarity --test_data RaindropClarity --num_worker $NUM_WORKER --print_freq $PRINT_FREQ --use_lpips $WANDB_FLAG"
COMMON_ITER="--num_iter $NUM_ITER --valid_freq $VALID_FREQ --lr_warmup_steps $LR_WARMUP_STEPS --crop $CROP_SIZE --batch_size $BATCH_SIZE --mixed_precision $MIXED_PRECISION"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session '$SESSION_NAME' already exists. Attach: tmux attach -t $SESSION_NAME"
  exit 0
fi

echo "tmux session '$SESSION_NAME' 생성 (ConvIR-base LR 스케줄 실험, 250k iter, GPU 0~2,5)..."

########################
# GPU 0: 실험 B - LR 1.5e-4, constant_with_warmup
########################
tmux new-session -d -s "$SESSION_NAME" -n "0-convir-base-lrB-const"
tmux send-keys -t "$SESSION_NAME:0" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+0)) main.py \
  --model_name ConvIR_base \
  --exp_suffix convir-base-lrB-const-250k \
  --learning_rate 1.5e-4 \
  --lr_scheduler constant_with_warmup \
  $COMMON_BASE \
  --lambda_l1 $LAMBDA_L1 --lambda_lpips $LAMBDA_LPIPS --lambda_msssim $LAMBDA_MSSSIM --lambda_fft $LAMBDA_FFT --use_ema \
  $COMMON_ITER --batch_size 2 --grad_accum 2" C-m

########################
# GPU 1: 실험 C - LR 1.5e-4, cosine
########################
tmux new-window -t "$SESSION_NAME" -n "1-convir-base-lrC-cosine"
tmux send-keys -t "$SESSION_NAME:1" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+1)) main.py \
  --model_name ConvIR_base \
  --exp_suffix convir-base-lrC-cosine-250k \
  --learning_rate 1.5e-4 \
  --lr_scheduler cosine \
  $COMMON_BASE \
  --lambda_l1 $LAMBDA_L1 --lambda_lpips $LAMBDA_LPIPS --lambda_msssim $LAMBDA_MSSSIM --lambda_fft $LAMBDA_FFT --use_ema \
  $COMMON_ITER --batch_size 2 --grad_accum 2" C-m

########################
# GPU 2: 실험 D - LR 1.5e-4, polynomial (linear decay)
########################
tmux new-window -t "$SESSION_NAME" -n "2-convir-base-lrD-poly"
tmux send-keys -t "$SESSION_NAME:2" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+2)) main.py \
  --model_name ConvIR_base \
  --exp_suffix convir-base-lrD-poly-250k \
  --learning_rate 1.5e-4 \
  --lr_scheduler polynomial --lr_power 1.0 \
  $COMMON_BASE \
  --lambda_l1 $LAMBDA_L1 --lambda_lpips $LAMBDA_LPIPS --lambda_msssim $LAMBDA_MSSSIM --lambda_fft $LAMBDA_FFT --use_ema \
  $COMMON_ITER --batch_size 2 --grad_accum 2" C-m

########################
# GPU 5: 실험 E - LR 1.5e-4, step_custom (3-step decay)
########################
tmux new-window -t "$SESSION_NAME" -n "5-convir-base-lrE-step"
tmux send-keys -t "$SESSION_NAME:5" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=5 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+5)) main.py \
  --model_name ConvIR_base \
  --exp_suffix convir-base-lrE-step-250k \
  --learning_rate 1.5e-4 \
  --lr_scheduler step_custom \
  $COMMON_BASE \
  --lambda_l1 $LAMBDA_L1 --lambda_lpips $LAMBDA_LPIPS --lambda_msssim $LAMBDA_MSSSIM --lambda_fft $LAMBDA_FFT --use_ema \
  $COMMON_ITER --batch_size 2 --grad_accum 2" C-m

echo "=========================================="
echo "Tmux session: $SESSION_NAME (ConvIR-base, 250k iter, GPU 0~2,5)"
echo "  0: 실험 B - constant_with_warmup, lr=1.5e-4"
echo "  1: 실험 C - cosine, lr=1.5e-4"
echo "  2: 실험 D - polynomial(linear), lr=1.5e-4"
echo "  5: 실험 E - step_custom(3-step decay), lr=1.5e-4"
echo "Loss (l1ema): lambda_l1=$LAMBDA_L1, lambda_lpips=$LAMBDA_LPIPS, lambda_msssim=$LAMBDA_MSSSIM, lambda_fft=$LAMBDA_FFT, EMA=on"
echo "=========================================="
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Switch: Ctrl+b then 0/1/2/5"
echo "=========================================="

