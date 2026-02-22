"""
Training script for Diffusion-based Image Restoration
Supports: DiffIR, SimpleDiffusion
"""
import os
import csv
import torch
import torch.nn.functional as F
from data import train_dataloader, valid_dataloader
from tqdm import tqdm
from diffusers.optimization import get_scheduler
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, ProjectConfiguration
from eval import psnr_y, ssim_y
import lpips as LPIPS
from torchvision.utils import make_grid
import datetime
import argparse


def train_diffusion(args):
    # Setup
    now = datetime.datetime.now()
    args.exp_name = f"{args.model_name}-{args.train_data}-{now.strftime('%m%d-%H%M')}"
    
    project_config = ProjectConfiguration(
        project_dir=f"experiment/{args.exp_name}",
        logging_dir=f"experiment/{args.exp_name}/logs"
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=project_config,
        gradient_accumulation_steps=args.grad_accum,
        kwargs_handlers=[ddp_kwargs]
    )
    
    # Initialize wandb (same project as train.py: NTIRE2026)
    if args.use_wandb and accelerator.is_main_process:
        import wandb
        wandb.init(project="NTIRE2026", name=args.exp_name, config=vars(args))
    
    set_seed(args.seed)
    device = accelerator.device
    
    # Model
    if args.model_name == 'DiffIR':
        from models.DiffIR import DiffIR
        model = DiffIR(
            cirp_dim=args.cirp_dim,
            cirp_blocks=args.cirp_blocks,
            dirt_dim=args.dirt_dim,
            num_timesteps=args.num_timesteps
        )
    else:  # SimpleDiffusion
        from models.SimpleDiffusion import SimpleDiffusion
        model = SimpleDiffusion(
            base_ch=args.base_ch,
            num_timesteps=args.num_timesteps
        )
    
    # Directories
    args.model_save_dir = os.path.join('results', f"{args.exp_name}", 'ckpt')
    args.log_dir = os.path.join('results', f"{args.exp_name}", 'logs')
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Save config
    if accelerator.is_main_process:
        with open(os.path.join(args.log_dir, 'config.txt'), 'w') as f:
            f.write(f'Model: {args.model_name}\n')
            f.write(f'Settings:\n')
            if args.model_name == 'DiffIR':
                f.write(f'  - cirp_dim: {args.cirp_dim}\n')
                f.write(f'  - cirp_blocks: {args.cirp_blocks}\n')
                f.write(f'  - dirt_dim: {args.dirt_dim}\n')
            else:
                f.write(f'  - base_ch: {args.base_ch}\n')
            f.write(f'  - num_timesteps: {args.num_timesteps}\n')
            f.write(f'  - num_inference_steps: {args.num_inference_steps}\n')
            f.write(f'  - batch_size: {args.batch_size}\n')
            f.write(f'  - crop_size: {args.crop}\n')
            f.write(f'  - learning_rate: {args.learning_rate}\n')
            f.write(f'  - lr_scheduler: {args.lr_scheduler}\n')
            f.write(f'  - num_iter: {args.num_iter}\n')
            f.write(f'  - mixed_precision: {args.mixed_precision}\n')
        
        # CSV 파일 초기화
        train_log_path = os.path.join(args.log_dir, 'train_log.csv')
        with open(train_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if args.model_name == 'DiffIR':
                writer.writerow(['iteration', 'total_loss', 'ir_loss', 'diff_loss', 'learning_rate'])
            else:
                writer.writerow(['iteration', 'loss', 'learning_rate'])
        
        valid_log_path = os.path.join(args.log_dir, 'valid_log.csv')
        with open(valid_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'psnr', 'ssim', 'lpips', 'best_psnr', 'best_ssim', 'best_lpips'])
    
    # Data
    dataloader = train_dataloader(
        f'data/train/{args.train_data}.txt',
        value_range=(0, 1),
        crop=args.crop,
        batch_size=args.batch_size,
        num_workers=args.num_worker
    )
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.num_iter
    )
    
    # Prepare
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    # LPIPS for validation
    lpips_fn = LPIPS.LPIPS(net='alex').to(device)
    lpips_fn.eval()
    
    # Training state
    global_iter = 0
    best_psnr = 0.0
    best_ssim = 0.0
    best_lpips = 1.0
    
    # Training loop
    pbar = tqdm(total=args.num_iter, desc=f'Training {args.model_name}')
    train_log = []
    valid_log = []
    
    model.train()
    while global_iter < args.num_iter:
        for data in dataloader:
            if global_iter >= args.num_iter:
                break
            
            degraded, clean = data
            
            with accelerator.accumulate(model):
                # Compute diffusion loss
                loss_output = accelerator.unwrap_model(model).training_loss(degraded, clean)
                
                # DiffIR returns (total, ir_loss, diff_loss), SimpleDiffusion returns single loss
                if isinstance(loss_output, tuple):
                    loss, ir_loss, diff_loss = loss_output
                else:
                    loss = loss_output
                    ir_loss = diff_loss = None
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            global_iter += 1
            pbar.update(1)
            
            # Logging
            if global_iter % args.print_freq == 0:
                lr = scheduler.get_last_lr()[0]
                log_entry = {
                    'iteration': global_iter,
                    'loss': loss.item(),
                    'learning_rate': lr
                }
                if ir_loss is not None:
                    log_entry['ir_loss'] = ir_loss.item()
                    log_entry['diff_loss'] = diff_loss.item()
                train_log.append(log_entry)
                
                # 실시간 CSV 저장 (main process only)
                if accelerator.is_main_process:
                    train_log_path = os.path.join(args.log_dir, 'train_log.csv')
                    with open(train_log_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        if ir_loss is not None:
                            writer.writerow([global_iter, loss.item(), ir_loss.item(), diff_loss.item(), lr])
                        else:
                            writer.writerow([global_iter, loss.item(), lr])
                
                # Wandb logging (main process only)
                if args.use_wandb and accelerator.is_main_process:
                    import wandb
                    wandb_log = {'Train/Loss': loss.item(), 'Train/LR': lr}
                    if ir_loss is not None:
                        wandb_log['Train/IR_Loss'] = ir_loss.item()
                        wandb_log['Train/Diff_Loss'] = diff_loss.item()
                    wandb.log(wandb_log, step=global_iter)
                
                if ir_loss is not None:
                    pbar.set_postfix(loss=f'{loss.item():.4f}', ir=f'{ir_loss.item():.4f}', diff=f'{diff_loss.item():.4f}')
                else:
                    pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{lr:.2e}')
            
            # Validation
            if global_iter % args.valid_freq == 0:
                # 모든 프로세스 동기화 (validation 전)
                accelerator.wait_for_everyone()
                
                if accelerator.is_main_process:
                    # Save model
                    save_name = os.path.join(args.model_save_dir, f'model_{global_iter}.pkl')
                    torch.save(accelerator.unwrap_model(model).state_dict(), save_name)
                    
                    # Validation
                    psnr, ssim, lpips_val = validate_diffusion(
                        accelerator.unwrap_model(model), 
                        device, args
                    )
                    
                    if psnr > best_psnr:
                        best_psnr = psnr
                    if ssim > best_ssim:
                        best_ssim = ssim
                    if lpips_val < best_lpips:
                        best_lpips = lpips_val
                    
                    valid_log.append({
                        'iteration': global_iter,
                        'psnr': psnr,
                        'ssim': ssim,
                        'lpips': lpips_val,
                        'best_psnr': best_psnr,
                        'best_ssim': best_ssim,
                        'best_lpips': best_lpips
                    })
                    
                    # 실시간 CSV 저장
                    valid_log_path = os.path.join(args.log_dir, 'valid_log.csv')
                    with open(valid_log_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([global_iter, psnr, ssim, lpips_val, best_psnr, best_ssim, best_lpips])
                    
                    # Wandb logging
                    if args.use_wandb:
                        import wandb
                        wandb.log({
                            'Valid/PSNR': psnr,
                            'Valid/SSIM': ssim,
                            'Valid/LPIPS': lpips_val,
                            'Valid/Best PSNR': best_psnr,
                            'Valid/Best SSIM': best_ssim,
                            'Valid/Best LPIPS': best_lpips
                        }, step=global_iter)
                    
                    print(f'\nIter {global_iter}: PSNR={psnr:.2f} (best={best_psnr:.2f}), '
                          f'SSIM={ssim:.4f}, LPIPS={lpips_val:.4f}')
                
                # 모든 프로세스 동기화 (validation 후)
                accelerator.wait_for_everyone()
                model.train()
    
    # Final save
    if accelerator.is_main_process:
        save_name = os.path.join(args.model_save_dir, 'Final.pkl')
        torch.save(accelerator.unwrap_model(model).state_dict(), save_name)
        
        # Save logs
        with open(os.path.join(args.log_dir, 'train_log.csv'), 'w', newline='') as f:
            if train_log:
                writer = csv.DictWriter(f, fieldnames=train_log[0].keys())
                writer.writeheader()
                writer.writerows(train_log)
        
        with open(os.path.join(args.log_dir, 'valid_log.csv'), 'w', newline='') as f:
            if valid_log:
                writer = csv.DictWriter(f, fieldnames=valid_log[0].keys())
                writer.writeheader()
                writer.writerows(valid_log)
    
    pbar.close()
    
    # End training
    if args.use_wandb:
        import wandb
        wandb.finish()
    
    print(f'\nTraining complete! Best PSNR: {best_psnr:.2f}')


@torch.no_grad()
def validate_diffusion(model, device, args):
    """Validate diffusion model"""
    model.eval()
    
    dataset = valid_dataloader(
        f'data/test/{args.test_data}.txt',
        value_range=(0, 1),
        paired=True,
        batch_size=1,
        num_workers=1
    )
    
    lpips_fn = LPIPS.LPIPS(net='alex').to(device)
    lpips_fn.eval()
    
    psnr_list = []
    ssim_list = []
    lpips_list = []

    # WandB 이미지 로깅 설정
    log_images = bool(getattr(args, 'use_wandb', False))
    max_log_images = int(getattr(args, 'wandb_num_images', 4))
    logged_images = 0
    wandb_samples = []
    
    for data in dataset:
        degraded, clean, name = data
        degraded, clean = degraded.to(device), clean.to(device)
        
        # Inference with DDIM
        pred = model(degraded, num_inference_steps=args.num_inference_steps)
        pred = pred.clamp(0, 1)
        
        # Metrics
        psnr_list.append(psnr_y(pred, clean).item())
        ssim_list.append(ssim_y(pred, clean).item())
        lpips_list.append(lpips_fn(pred * 2 - 1, clean * 2 - 1).item())

        # WandB: 입력/예측/정답을 한 장으로 묶어서 기록
        if log_images and logged_images < max_log_images:
            # [1,3,H,W] 3장을 가로로 concat
            triplet = torch.cat([degraded, pred, clean], dim=3)  # [1,3,H,3W]
            # make_grid는 [C,H,W]로 만들기 위해 batch 차원 제거
            grid = make_grid(triplet, nrow=1)  # [3,H,3W]
            wandb_samples.append((grid.detach().cpu(), name[0]))
            logged_images += 1

    if log_images and wandb_samples:
        import wandb
        wandb.log(
            {
                "Valid/Samples": [
                    wandb.Image(img, caption=f"{fname} | left:input mid:pred right:gt")
                    for (img, fname) in wandb_samples
                ]
            }
        )
    
    return sum(psnr_list) / len(psnr_list), \
           sum(ssim_list) / len(ssim_list), \
           sum(lpips_list) / len(lpips_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--train_data', type=str, default='RaindropClarity')
    parser.add_argument('--test_data', type=str, default='RaindropClarity')
    parser.add_argument('--seed', type=int, default=42)
    
    # Model selection
    parser.add_argument('--model_name', type=str, default='DiffIR', choices=['DiffIR', 'SimpleDiffusion'])
    
    # SimpleDiffusion params
    parser.add_argument('--base_ch', type=int, default=64)
    
    # DiffIR params
    parser.add_argument('--cirp_dim', type=int, default=64)
    parser.add_argument('--cirp_blocks', type=int, default=8)
    parser.add_argument('--dirt_dim', type=int, default=64)
    
    # Common diffusion params
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--num_inference_steps', type=int, default=20)
    
    # Training
    parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--crop', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler', type=str, default='cosine')
    parser.add_argument('--lr_warmup_steps', type=int, default=500)
    parser.add_argument('--num_iter', type=int, default=50000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--valid_freq', type=int, default=5000)
    parser.add_argument('--num_worker', type=int, default=4)
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_num_images', type=int, default=4, help='Number of validation sample images to log to wandb')
    
    args = parser.parse_args()
    print(args)
    
    train_diffusion(args)
