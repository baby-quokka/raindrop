
CUDA_VISIBLE_DEVICES=0,1 accelerate launch main.py --mode train \
  --model_name Restormer \
  --train_data RaindropClarity \
  --num_iter 500 \
  --batch_size 2 \
  --print_freq 50 \
  --valid_freq 500
