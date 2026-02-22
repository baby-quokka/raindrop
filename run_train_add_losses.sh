#!/bin/bash
#
# Stage1 (Drop → Clear) 8실험 병렬 학습 (GPU 번호 = tmux 창 번호)
# - tmux 창 0~7 = GPU 0~7, 포트 29500~29507
#   0,1: LPIPS 0.05 (NAFNet, ConvIR)
#   2,3: LPIPS 0.02 (NAFNet, ConvIR)
#   4,5: 원래 loss 0.1 (NAFNet, ConvIR)
#   6,7: LPIPS 0.05 + EMA (NAFNet, ConvIR)
# results 폴더: NAFNet-05-RaindropClarity-*, ConvIR-05-*, NAFNet-02-*, ConvIR-02-*,
#               NAFNet-ema-*, ConvIR-ema-*, NAFNet-05ema-*, ConvIR-05ema-* (끝에 일시 추가)
#
# 사용법:
#   bash run_train_add_losses.sh              # 세션명 기본값: train-add-losses
#   bash run_train_add_losses.sh mysession    # 세션명 지정
#

SESSION_NAME="${1:-train-add-losses}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

#
# conda 환경 자동 활성화
# - 기본 env 이름: raindrop  (CONDA_ENV로 오버라이드 가능)
#
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
  echo "Warning: conda base not found. Run without 'conda activate'."
  PRE_CMD=""
fi

# ==========================
# GPU / PORT (창 번호 = GPU 번호, 포트 29500~29507)
# ==========================
PORT_BASE="${PORT_BASE:-29500}"

# ==========================
# 하이퍼파라미터 기본값
# ==========================
NUM_ITER="${NUM_ITER:-50000}"
VALID_FREQ="${VALID_FREQ:-10000}"
BATCH_SIZE="${BATCH_SIZE:-4}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-constant_with_warmup}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-5000}"

CROP_SIZE="${CROP_SIZE:-256}"
NUM_WORKER="${NUM_WORKER:-4}"
PRINT_FREQ="${PRINT_FREQ:-100}"

MIXED_PRECISION="${MIXED_PRECISION:-no}"

# WandB 사용 여부 (기본 ON)
USE_WANDB="${USE_WANDB:-1}"
WANDB_FLAG=""
if [ "$USE_WANDB" = "1" ]; then
  WANDB_FLAG="--use_wandb"
fi

# Loss 비율 (공통)
LAMBDA_MSSSIM="${LAMBDA_MSSSIM:-0.1}"
LAMBDA_FFT="${LAMBDA_FFT:-0.01}"
# lambda_lpips: 0.05(창 0,1,6,7), 0.02(창 2,3), 0.1 원래(창 4,5)
LAMBDA_LPIPS_05="${LAMBDA_LPIPS_05:-0.05}"
LAMBDA_LPIPS_02="${LAMBDA_LPIPS_02:-0.02}"
LAMBDA_LPIPS_ORIG="${LAMBDA_LPIPS_ORIG:-0.1}"

# 공통 인자 (모델/배치/iter 등만, loss/EMA는 창별로 다름)
COMMON="--train_data RaindropClarity --test_data RaindropClarity --learning_rate $LEARNING_RATE --lr_scheduler $LR_SCHEDULER --num_worker $NUM_WORKER --print_freq $PRINT_FREQ --use_lpips --lambda_msssim $LAMBDA_MSSSIM --lambda_fft $LAMBDA_FFT $WANDB_FLAG"
COMMON_ITER="--num_iter $NUM_ITER --valid_freq $VALID_FREQ --lr_warmup_steps $LR_WARMUP_STEPS --crop $CROP_SIZE --batch_size $BATCH_SIZE --mixed_precision $MIXED_PRECISION"

# 이미 세션이 있으면 붙기만 하라고 안내하고 종료
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session '$SESSION_NAME' already exists. Attach with: tmux attach -t $SESSION_NAME"
  exit 0
fi

echo "tmux session '$SESSION_NAME' 생성 및 학습 시작..."

# 0,1: LPIPS 0.05 → results/NAFNet-05-RaindropClarity-..., ConvIR-05-RaindropClarity-...
tmux new-session -d -s "$SESSION_NAME" -n "0-nafnet-lp05"
tmux send-keys -t "$SESSION_NAME:0" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+0)) main.py --model_name NAFNet --exp_suffix 05 $COMMON --lambda_lpips $LAMBDA_LPIPS_05 $COMMON_ITER" C-m

tmux new-window -t "$SESSION_NAME" -n "1-convir-lp05"
tmux send-keys -t "$SESSION_NAME:1" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+1)) main.py --model_name ConvIR --exp_suffix 05 $COMMON --lambda_lpips $LAMBDA_LPIPS_05 $COMMON_ITER" C-m

# 2,3: LPIPS 0.02 → results/NAFNet-02-RaindropClarity-..., ConvIR-02-RaindropClarity-...
tmux new-window -t "$SESSION_NAME" -n "2-nafnet-lp02"
tmux send-keys -t "$SESSION_NAME:2" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+2)) main.py --model_name NAFNet --exp_suffix 02 $COMMON --lambda_lpips $LAMBDA_LPIPS_02 $COMMON_ITER" C-m

tmux new-window -t "$SESSION_NAME" -n "3-convir-lp02"
tmux send-keys -t "$SESSION_NAME:3" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+3)) main.py --model_name ConvIR --exp_suffix 02 $COMMON --lambda_lpips $LAMBDA_LPIPS_02 $COMMON_ITER" C-m

# 4,5: 원래 loss 0.1 → results/NAFNet-ema-RaindropClarity-..., ConvIR-ema-RaindropClarity-...
tmux new-window -t "$SESSION_NAME" -n "4-nafnet-orig"
tmux send-keys -t "$SESSION_NAME:4" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=4 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+4)) main.py --model_name NAFNet --exp_suffix ema $COMMON --lambda_lpips $LAMBDA_LPIPS_ORIG $COMMON_ITER" C-m

tmux new-window -t "$SESSION_NAME" -n "5-convir-orig"
tmux send-keys -t "$SESSION_NAME:5" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=5 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+5)) main.py --model_name ConvIR --exp_suffix ema $COMMON --lambda_lpips $LAMBDA_LPIPS_ORIG $COMMON_ITER" C-m

# 6,7: LPIPS 0.05 + EMA → results/NAFNet-05ema-RaindropClarity-..., ConvIR-05ema-RaindropClarity-...
tmux new-window -t "$SESSION_NAME" -n "6-nafnet-lp05-ema"
tmux send-keys -t "$SESSION_NAME:6" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=6 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+6)) main.py --model_name NAFNet --exp_suffix 05ema $COMMON --lambda_lpips $LAMBDA_LPIPS_05 --use_ema $COMMON_ITER" C-m

tmux new-window -t "$SESSION_NAME" -n "7-convir-lp05-ema"
tmux send-keys -t "$SESSION_NAME:7" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=7 accelerate launch --num_processes 1 --main_process_port $((PORT_BASE+7)) main.py --model_name ConvIR --exp_suffix 05ema $COMMON --lambda_lpips $LAMBDA_LPIPS_05 --use_ema $COMMON_ITER" C-m

echo "=========================================="
echo "Tmux session: $SESSION_NAME (창 번호 = GPU 번호, 포트 $PORT_BASE~$((PORT_BASE+7)))"
echo "  0: NAFNet  LPIPS 0.05           GPU 0, port $((PORT_BASE+0))"
echo "  1: ConvIR  LPIPS 0.05           GPU 1, port $((PORT_BASE+1))"
echo "  2: NAFNet  LPIPS 0.02           GPU 2, port $((PORT_BASE+2))"
echo "  3: ConvIR  LPIPS 0.02           GPU 3, port $((PORT_BASE+3))"
echo "  4: NAFNet  원래 loss 0.1        GPU 4, port $((PORT_BASE+4))"
echo "  5: ConvIR  원래 loss 0.1        GPU 5, port $((PORT_BASE+5))"
echo "  6: NAFNet  LPIPS 0.05 + EMA     GPU 6, port $((PORT_BASE+6))"
echo "  7: ConvIR  LPIPS 0.05 + EMA     GPU 7, port $((PORT_BASE+7))"
echo "=========================================="
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Switch: Ctrl+b then 0~7"
echo "=========================================="
