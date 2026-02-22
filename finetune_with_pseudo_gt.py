"""
B안: Pseudo-GT를 사용한 파인튜닝 스크립트

STRRNet 권장 implementation detail (성능 안정화용):
  - 10k iter, LR=1e-4 (논문), geometric augmentation + mixup 비활성화.
  - 우리 기본값: LR=1e-5. STRRNet 스타일로 재시도 시:
    --learning_rate 1e-4 --no_augment

사용법:
    1. 먼저 pseudo-GT 생성:
        python generate_pseudo_gt.py ...
    2. 파인튜닝 실행:
        python finetune_with_pseudo_gt.py \
            --checkpoint ... --pseudo_gt_dir pseudo_gt_output \
            --input_list data/valid/RaindropClarity.txt \
            --num_iter 10000 --learning_rate 1e-5
        # STRRNet 스타일: --no_augment --learning_rate 1e-4
"""

import os
import csv
import json
import types
import copy
import datetime
import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm
import lpips as LPIPS

# XPU 호환성 처리
if not hasattr(torch, "xpu"):
    torch.xpu = types.SimpleNamespace(empty_cache=lambda: None)

from diffusers.optimization import get_scheduler
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, ProjectConfiguration

import models
from data.pseudo_gt_dataset import pseudo_gt_train_dataloader
from valid import _valid
from train import (
    compute_ms_ssim_loss,
    compute_fft_loss,
    get_semantic_labels,
)


def finetune_with_pseudo_gt(model, accelerator, args):
    """
    Pseudo-GT를 사용한 파인튜닝
    
    Args:
        model: 사전 학습된 모델
        accelerator: Accelerator 객체
        args: 학습 인자
    """
    device = accelerator.device
    
    # 로그 디렉토리 설정
    args.log_dir = os.path.join(args.exp_dir, "logs")
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 학습 로그 초기화
    if accelerator.is_main_process:
        train_log_path = os.path.join(args.log_dir, 'train_log.csv')
        with open(train_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'l1_loss', 'lpips_loss', 'msssim_loss', 'fft_loss', 'total_loss', 'learning_rate'])
    
    # Loss 함수 설정
    if args.use_lpips:
        lpips_fn = LPIPS.LPIPS(net='vgg').to(device)
        lpips_fn.requires_grad_(False), lpips_fn.eval()
    
    # Pseudo-GT 데이터로더 생성 (STRRNet 권장: no_augment 시 geometric aug 비활성화)
    use_augment = not getattr(args, "no_augment", False)
    dataloader = pseudo_gt_train_dataloader(
        input_list=args.input_list,
        pseudo_gt_dir=args.pseudo_gt_dir,
        crop=args.crop,
        value_range=args.value_range,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        use_augment=use_augment,
    )
    
    model, dataloader = accelerator.prepare(model, dataloader)
    
    # Optimizer 및 Scheduler 설정
    # 파인튜닝은 작은 learning rate 사용
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    scheduler_name = args.lr_scheduler
    if scheduler_name == "step_custom":
        scheduler_name = "constant"
    
    scheduler = get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.num_iter,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power
    )
    
    # EMA 설정
    use_ema = getattr(args, 'use_ema', False)
    ema_decay = getattr(args, 'ema_decay', 0.999)
    ema_state = None
    if use_ema:
        ema_state = copy.deepcopy(accelerator.unwrap_model(model).state_dict())
        if accelerator.is_main_process:
            print(f"EMA enabled (decay={ema_decay})")
    
    # 체크포인트 저장 디렉토리
    ckpt_dir = os.path.join(args.exp_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    model.train()
    
    pbar = tqdm(range(args.num_iter), disable=not accelerator.is_main_process, desc="Finetuning")
    iter_loader = iter(dataloader)
    global_iter = 0
    iter_l1_adder = iter_lpips_adder = 0.0
    iter_msssim_adder = iter_fft_adder = 0.0
    iter_loss_adder = 0.0
    iter_adder = 0
    
    while global_iter < args.num_iter:
        batch = next(iter_loader, None)
        if batch is None:
            iter_loader = iter(dataloader)
            batch = next(iter_loader, None)
        
        input_img, pseudo_gt = batch
        input_img = input_img.to(device, non_blocking=True)
        pseudo_gt = pseudo_gt.to(device, non_blocking=True)
        
        with accelerator.accumulate(model):
            with accelerator.autocast():
                # 모델 forward
                if hasattr(model, 'use_semantic_guidance') and model.use_semantic_guidance:
                    semantic_labels = get_semantic_labels(input_img, pseudo_gt)
                    pred_img = model(input_img, semantic_labels=semantic_labels)
                else:
                    pred_img = model(input_img)
                
                if isinstance(pred_img, (list, tuple)):
                    pred_img_list = pred_img
                    pred_img = pred_img[-1]
                else:
                    pred_img_list = None
                
                # Loss 계산
                l1_loss = F.l1_loss(pred_img, pseudo_gt)
                
                total_loss = args.lambda_l1 * l1_loss
                
                # LPIPS loss
                if args.use_lpips:
                    lpips_loss = lpips_fn(pred_img, pseudo_gt).mean()
                    total_loss = total_loss + args.lambda_lpips * lpips_loss
                else:
                    lpips_loss = torch.tensor(0.0, device=device)
                
                # MS-SSIM loss
                if args.lambda_msssim > 0:
                    msssim_loss = compute_ms_ssim_loss(pred_img, pseudo_gt, data_range=args.value_range)
                    total_loss = total_loss + args.lambda_msssim * msssim_loss
                else:
                    msssim_loss = torch.tensor(0.0, device=device)
                
                # FFT loss
                if args.lambda_fft > 0:
                    fft_loss = compute_fft_loss(pred_img, pseudo_gt)
                    total_loss = total_loss + args.lambda_fft * fft_loss
                else:
                    fft_loss = torch.tensor(0.0, device=device)
            
            accelerator.backward(total_loss)
            
            if accelerator.sync_gradients:
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # EMA 업데이트
            if use_ema and accelerator.sync_gradients:
                with torch.no_grad():
                    for k, v in accelerator.unwrap_model(model).state_dict().items():
                        if k in ema_state:
                            ema_state[k] = ema_decay * ema_state[k] + (1 - ema_decay) * v
            
            if accelerator.sync_gradients:
                iter_l1_adder += l1_loss.item()
                iter_lpips_adder += lpips_loss.item()
                iter_msssim_adder += msssim_loss.item()
                iter_fft_adder += fft_loss.item()
                iter_loss_adder += total_loss.item()
                iter_adder += 1
                
                if global_iter % args.print_freq == 0 and iter_adder > 0:
                    avg_l1 = iter_l1_adder / iter_adder
                    avg_lpips = iter_lpips_adder / iter_adder
                    avg_msssim = iter_msssim_adder / iter_adder
                    avg_fft = iter_fft_adder / iter_adder
                    avg_loss = iter_loss_adder / iter_adder
                    lr = scheduler.get_last_lr()[0]
                    
                    pbar.set_postfix({
                        'L1': f'{avg_l1:.4f}',
                        'LPIPS': f'{avg_lpips:.4f}',
                        'MS-SSIM': f'{avg_msssim:.4f}',
                        'FFT': f'{avg_fft:.4f}',
                        'Loss': f'{avg_loss:.4f}',
                        'LR': f'{lr:.2e}'
                    })
                    
                    if accelerator.is_main_process:
                        train_log_path = os.path.join(args.log_dir, 'train_log.csv')
                        with open(train_log_path, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                global_iter, avg_l1, avg_lpips, avg_msssim, avg_fft,
                                avg_loss, lr
                            ])
                    
                    iter_l1_adder = iter_lpips_adder = 0.0
                    iter_msssim_adder = iter_fft_adder = 0.0
                    iter_loss_adder = 0.0
                    iter_adder = 0
                
                # 체크포인트 저장
                if (global_iter + 1) % args.valid_freq == 0:
                    if accelerator.is_main_process:
                        # EMA 가중치로 검증
                        if use_ema:
                            original_state = accelerator.unwrap_model(model).state_dict()
                            accelerator.unwrap_model(model).load_state_dict(ema_state)
                        
                        # 검증 (간단히 loss만 기록)
                        model.eval()
                        val_loss = 0.0
                        val_count = 0
                        with torch.no_grad():
                            val_iter = iter(dataloader)
                            for _ in range(min(100, len(dataloader))):  # 샘플링
                                try:
                                    val_input, val_gt = next(val_iter)
                                    val_input = val_input.to(device)
                                    val_gt = val_gt.to(device)
                                    
                                    if hasattr(model, 'use_semantic_guidance') and model.use_semantic_guidance:
                                        val_semantic = get_semantic_labels(val_input, val_gt)
                                        val_pred = model(val_input, semantic_labels=val_semantic)
                                    else:
                                        val_pred = model(val_input)
                                    
                                    if isinstance(val_pred, (list, tuple)):
                                        val_pred = val_pred[-1]
                                    
                                    val_loss += F.l1_loss(val_pred, val_gt).item()
                                    val_count += 1
                                except StopIteration:
                                    break
                        
                        val_loss = val_loss / val_count if val_count > 0 else 0.0
                        print(f"\n[Iter {global_iter+1}] Validation L1 Loss: {val_loss:.4f}")
                        
                        # 체크포인트 저장
                        save_dict = {
                            'model': accelerator.unwrap_model(model).state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'iteration': global_iter + 1,
                            'val_loss': val_loss,
                        }
                        if use_ema:
                            save_dict['ema_state'] = ema_state
                        
                        torch.save(save_dict, os.path.join(ckpt_dir, f"iter_{global_iter+1}.pkl"))
                        
                        # 최종 체크포인트도 저장
                        torch.save(save_dict, os.path.join(ckpt_dir, "Final.pkl"))
                        
                        # EMA 복원
                        if use_ema:
                            accelerator.unwrap_model(model).load_state_dict(original_state)
                        
                        model.train()
                
                global_iter += 1
                pbar.update(1)
    
    pbar.close()
    print(f"\n[Info] Finetuning completed! Final checkpoint saved to: {os.path.join(ckpt_dir, 'Final.pkl')}")


def main():
    parser = argparse.ArgumentParser(description="Finetune model with pseudo-GT")
    
    # 모델 및 체크포인트
    parser.add_argument("--model_name", type=str, default="Restormer",
                        help="모델 이름")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="사전 학습된 모델 체크포인트 경로")
    
    # 데이터
    parser.add_argument("--input_list", type=str, required=True,
                        help="입력 이미지 리스트 txt 경로")
    parser.add_argument("--pseudo_gt_dir", type=str, required=True,
                        help="Pseudo-GT 이미지가 저장된 디렉토리")
    
    # 실험 설정
    parser.add_argument("--exp_suffix", type=str, default="finetune_pseudo_gt",
                        help="실험 이름 suffix")
    parser.add_argument("--exp_dir", type=str, default=None,
                        help="실험 디렉토리 (지정하지 않으면 자동 생성)")
    
    # 학습 하이퍼파라미터
    parser.add_argument("--num_iter", type=int, default=10000,
                        help="파인튜닝 iteration 수")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate. STRRNet은 1e-4 사용 (--no_augment와 함께 시도 권장)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--crop", type=int, default=256,
                        help="Crop size")
    parser.add_argument("--value_range", type=tuple, default=(0, 1),
                        help="값 범위")
    parser.add_argument("--num_worker", type=int, default=4,
                        help="DataLoader worker 수")
    parser.add_argument("--no_augment", action="store_true",
                        help="STRRNet 권장: geometric aug 비활성화 (center crop만, flip 없음). pseudo-GT 노이즈 학습 완화.")
    
    # Loss 가중치
    parser.add_argument("--lambda_l1", type=float, default=1.0,
                        help="L1 loss 가중치")
    parser.add_argument("--lambda_lpips", type=float, default=0.1,
                        help="LPIPS loss 가중치")
    parser.add_argument("--lambda_msssim", type=float, default=0.1,
                        help="MS-SSIM loss 가중치")
    parser.add_argument("--lambda_fft", type=float, default=0.01,
                        help="FFT loss 가중치")
    parser.add_argument("--use_lpips", action="store_true",
                        help="LPIPS loss 사용")
    
    # 기타
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help="Learning rate scheduler")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="LR warmup steps")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                        help="Number of cycles for cosine_with_restarts scheduler")
    parser.add_argument("--lr_power", type=float, default=1.0,
                        help="Power factor for polynomial scheduler")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping")
    parser.add_argument("--print_freq", type=int, default=100,
                        help="Print frequency")
    parser.add_argument("--valid_freq", type=int, default=1000,
                        help="Validation/checkpoint frequency")
    parser.add_argument("--use_ema", action="store_true",
                        help="EMA 사용")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps")
    
    args = parser.parse_args()
    
    # 실험 디렉토리 설정
    if args.exp_dir is None:
        now = datetime.datetime.now()
        args.exp_dir = f"experiment/{args.model_name}-{args.exp_suffix}-{now.strftime('%m%d-%H%M')}"
    
    os.makedirs(args.exp_dir, exist_ok=True)
    
    # 설정 저장
    with open(os.path.join(args.exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Accelerator 설정
    project_config = ProjectConfiguration(
        project_dir=args.exp_dir,
        logging_dir=os.path.join(args.exp_dir, "logs")
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=project_config,
        gradient_accumulation_steps=args.grad_accum,
        kwargs_handlers=[ddp_kwargs]
    )
    
    set_seed(args.seed)
    
    # 모델 로드
    model_class = getattr(models, args.model_name)
    model = model_class()
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    print(f"[Info] Loaded checkpoint from: {args.checkpoint}")
    print(f"[Info] Experiment directory: {args.exp_dir}")
    
    # 파인튜닝 실행
    finetune_with_pseudo_gt(model, accelerator, args)


if __name__ == "__main__":
    main()
