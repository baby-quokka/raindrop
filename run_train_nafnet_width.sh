#!/bin/bash
#
# NAFNet width 실험: width48 (GPU 6), width64 (GPU 7)
# tmux 창 3개: 0=NAFNet_width48, 1=NAFNet_width64, 2=빈 창
#
# 사용법:
#   bash run_train_nafnet_width.sh
#   bash run_train_nafnet_width.sh mysession
#

SESSION_NAME="${1:-train-nafnet-width}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# conda 활성화
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  PRE_CMD="source /opt/conda/etc/profile.d/conda.sh && conda activate raindrop && "
else
  PRE_CMD=""
fi

# 공통 인자 (예전 학습과 동일)
COMMON="--train_data RaindropClarity --test_data RaindropClarity --learning_rate 1e-4 --lr_scheduler constant_with_warmup --num_worker 4 --print_freq 100 --use_lpips --lambda_msssim 0.1 --lambda_fft 0.01 --use_wandb --lambda_lpips 0.1 --num_iter 50000 --valid_freq 10000 --lr_warmup_steps 5000 --crop 256 --batch_size 4 --mixed_precision no"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Session '$SESSION_NAME' already exists. Attach with: tmux attach -t $SESSION_NAME"
  exit 0
fi

echo "tmux session '$SESSION_NAME' 생성..."

# 창 0: NAFNet_width48 on GPU 6, port 29500
tmux new-session -d -s "$SESSION_NAME" -n "0-nafnet-width48"
tmux send-keys -t "$SESSION_NAME:0" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=6 accelerate launch --num_processes 1 --main_process_port 29500 main.py --model_name NAFNet_width48 --exp_suffix width48 $COMMON" C-m

# 창 1: NAFNet_width64 on GPU 7, port 29501
tmux new-window -t "$SESSION_NAME" -n "1-nafnet-width64"
tmux send-keys -t "$SESSION_NAME:1" "${PRE_CMD}cd $SCRIPT_DIR && CUDA_VISIBLE_DEVICES=7 accelerate launch --num_processes 1 --main_process_port 29501 main.py --model_name NAFNet_width64 --exp_suffix width64 $COMMON" C-m

# 창 2: 빈 창
tmux new-window -t "$SESSION_NAME" -n "2-empty"

echo "=========================================="
echo "Tmux session: $SESSION_NAME"
echo "  0: NAFNet_width48  GPU 6, port 29500"
echo "  1: NAFNet_width64  GPU 7, port 29501"
echo "  2: (empty)"
echo "=========================================="
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Switch: Ctrl+b then 0, 1, 2"
echo "=========================================="
