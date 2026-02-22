#!/bin/bash
#
# 장기 학습 200k iter, 8실험 병렬 (GPU 0~7)
# - iter 5만 → 20만
# - OOM 우려 없음: iter 늘리는 건 step 수만 늘리는 것
#
# 0,1: 기존 loss (lambda 0.1) NAFNet, ConvIR
# 2,3: LPIPS 0.05 NAFNet, ConvIR
# 4,5: 기존 loss + EMA NAFNet, ConvIR
# 6,7: L1 가중치 변경(1.5) + EMA NAFNet, ConvIR
#
# 사용법:
#   bash run_train_long.sh
#   bash run_train_long.sh train-long
#

SESSION_NAME="${1:-train-long}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

# 200k iter, valid 10번
NUM_ITER="${NUM_ITER:-200000}"
VALID_FREQ="${VALID_FREQ:-20000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-5000}"
CROP_SIZE="${CROP_SIZE:-256}"
NUM_WORKER="${NUM_WORKER:-4}"
PRINT_FREQ="${PRINT_FREQ:-100}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
PORT_BASE="${PORT_BASE:-29600}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_FLAG=""
[ "$USE_WANDB" = "1" ] && WANDB_FLAG="--use_wandb"

LAMBDA_LPIPS_ORIG="${LAMBDA_LPIPS_ORIG:-0.1}"
LAMBDA_LPIPS_05="${LAMBDA_LPIPS_05:-0.05}"
LAMBDA_MSSSIM="${LAMBDA_MSSSIM:-0.1}"
LAMBDA_FFT="${LAMBDA_FFT:-0.01}"
LAMBDA_L1_BOOST="${LAMBDA_L1_BOOST:-1.5}"

COMMON="--train_data RaindropClarity --test_data RaindropClarity --learning_rate 1e-4 --lr_scheduler constant_with_warmup --num_worker $NUM_WORKER --print_freq $PRINT_FREQ --use_lpips --lambda_msssim $LAMBDA_MSSSIM --lambda_fft $LAMBDA_FFT $WANDB_FLAG"
COMMON_ITER="--num_iter $NUM_ITER --valid_freq $VALID_FREQ --lr_warmup_steps $LR_WARMUP_STEPS --crop $CROP_SIZE --batch_size $BATCH_SIZE --mixed_precision $MIXED_PRECISION"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session '$SESSION_NAME' already exists. Attach: tmux attach -t $SESSION_NAME"
  exit 0
fi

echo "tmux session '$SESSION_NAME' 생성 (200k iter, GPU 0~7)..."

# 0,1: 기존 loss (NAFNet, ConvIR)
tmux new-session -d -s "$SESSION_NAME" -n "0-nafnet-orig"
tmux send-keys -t "$SESSION_NAME:0" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+0)) main.py --model_name NAFNet --exp_suffix orig200k $COMMON --lambda_lpips $LAMBDA_LPIPS_ORIG $COMMON_ITER" C-m

tmux new-window -t "$SESSION_NAME" -n "1-convir-orig"
tmux send-keys -t "$SESSION_NAME:1" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+1)) main.py --model_name ConvIR --exp_suffix orig200k $COMMON --lambda_lpips $LAMBDA_LPIPS_ORIG $COMMON_ITER" C-m

# 2,3: LPIPS 0.05 (NAFNet, ConvIR)
tmux new-window -t "$SESSION_NAME" -n "2-nafnet-lp05"
tmux send-keys -t "$SESSION_NAME:2" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+2)) main.py --model_name NAFNet --exp_suffix 05-200k $COMMON --lambda_lpips $LAMBDA_LPIPS_05 $COMMON_ITER" C-m

tmux new-window -t "$SESSION_NAME" -n "3-convir-lp05"
tmux send-keys -t "$SESSION_NAME:3" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+3)) main.py --model_name ConvIR --exp_suffix 05-200k $COMMON --lambda_lpips $LAMBDA_LPIPS_05 $COMMON_ITER" C-m

# 4,5: 기존 loss + EMA (NAFNet, ConvIR)
tmux new-window -t "$SESSION_NAME" -n "4-nafnet-ema"
tmux send-keys -t "$SESSION_NAME:4" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=4 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+4)) main.py --model_name NAFNet --exp_suffix ema200k $COMMON --lambda_lpips $LAMBDA_LPIPS_ORIG --use_ema $COMMON_ITER" C-m

tmux new-window -t "$SESSION_NAME" -n "5-convir-ema"
tmux send-keys -t "$SESSION_NAME:5" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=5 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+5)) main.py --model_name ConvIR --exp_suffix ema200k $COMMON --lambda_lpips $LAMBDA_LPIPS_ORIG --use_ema $COMMON_ITER" C-m

# 6,7: L1 가중치 1.5배 + EMA (NAFNet, ConvIR)
tmux new-window -t "$SESSION_NAME" -n "6-nafnet-l1ema"
tmux send-keys -t "$SESSION_NAME:6" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=6 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+6)) main.py --model_name NAFNet --exp_suffix l1ema200k $COMMON --lambda_l1 $LAMBDA_L1_BOOST --lambda_lpips $LAMBDA_LPIPS_ORIG --use_ema $COMMON_ITER" C-m

tmux new-window -t "$SESSION_NAME" -n "7-convir-l1ema"
tmux send-keys -t "$SESSION_NAME:7" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=7 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+7)) main.py --model_name ConvIR --exp_suffix l1ema200k $COMMON --lambda_l1 $LAMBDA_L1_BOOST --lambda_lpips $LAMBDA_LPIPS_ORIG --use_ema $COMMON_ITER" C-m

echo "=========================================="
echo "Tmux: $SESSION_NAME | iter=$NUM_ITER | 포트 $PORT_BASE~$((PORT_BASE+7))"
echo "  0: NAFNet  기존 loss       GPU 0"
echo "  1: ConvIR  기존 loss       GPU 1"
echo "  2: NAFNet  LPIPS 0.05      GPU 2"
echo "  3: ConvIR  LPIPS 0.05      GPU 3"
echo "  4: NAFNet  기존+EMA        GPU 4"
echo "  5: ConvIR  기존+EMA        GPU 5"
echo "  6: NAFNet  L1($LAMBDA_L1_BOOST)+EMA GPU 6"
echo "  7: ConvIR  L1($LAMBDA_L1_BOOST)+EMA GPU 7"
echo "=========================================="
echo "Attach: tmux attach -t $SESSION_NAME"
echo "=========================================="
