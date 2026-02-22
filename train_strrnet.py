"""
STRRNet Training Script
NTIRE 2025 Challenge on Day and Night Raindrop Removal - 1st Place (Miracle Team)

Training Configuration (from paper):
- Optimizer: Adam
- Total iterations: 500,000
- Learning rate: 0.0003 (first 92,000 iters) → decay to 0.000001
- Patch size: 128x128
- Batch size: variable (paper used RTX 4090)
- Loss: L1 (weight=1) + MS-SSIM (weight=0.2)
- Data augmentation: geometric (rotation, flip)

Two-stage training:
1. Pre-training on original dataset
2. Semi-supervised fine-tuning with median fusion (optional)
"""

import os
import csv
import json
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from accelerate import Accelerator

from models.STRRNet import STRRNet, SemanticClassifier
from data.strrnet_dataset import STRRNetDataset, strrnet_train_dataloader
from losses.ms_ssim import MSSSIMLoss
from valid import _valid


def get_strrnet_lr_scheduler(optimizer, num_iter, warmup_iters=92000):
    """
    STRRNet learning rate scheduler:
    - First 92,000 iterations: constant 0.0003
    - Remaining iterations: cosine decay from 0.0003 to 0.000001
    """
    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            # Constant learning rate phase
            return 1.0
        else:
            # Cosine decay phase
            progress = (current_iter - warmup_iters) / (num_iter - warmup_iters)
            # Decay from 1.0 to 0.000001/0.0003 = 0.00333
            min_lr_ratio = 0.000001 / 0.0003
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_strrnet(model, accelerator, args):
    """Main training loop for STRRNet."""
    device = accelerator.device
    use_wandb = getattr(args, 'use_wandb', False)
    
    if use_wandb:
        import wandb
    
    # Generate run name with timestamp (used for both wandb and local folders)
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d-%H%M")
    run_name = f"STRRNet-{args.train_data}-{timestamp}"
    args.exp_name = run_name  # Update exp_name to match wandb
    
    # Setup directories and logging
    if accelerator.is_main_process:
        if use_wandb:
            wandb.init(project="NTIRE2026", name=run_name, config=vars(args))
        
        args.model_save_dir = os.path.join('results', f"{args.exp_name}", 'ckpt')
        args.log_dir = os.path.join('results', f"{args.exp_name}", 'logs')
        os.makedirs(args.model_save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        
        # Save config
        config_path = os.path.join(args.log_dir, 'config.json')
        config_dict = {}
        for key, value in vars(args).items():
            if isinstance(value, tuple):
                config_dict[key] = list(value)
            else:
                config_dict[key] = value
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        print(f'Config saved to: {config_path}')
        
        # Save readable config
        config_txt_path = os.path.join(args.log_dir, 'config.txt')
        with open(config_txt_path, 'w', encoding='utf-8') as f:
            f.write(f'STRRNet Training - {args.exp_name}\n')
            f.write('=' * 60 + '\n')
            f.write(f'Model: STRRNet\n')
            f.write(f'Train Data: {args.train_data}\n')
            f.write(f'Test Data: {args.test_data}\n')
            f.write('=' * 60 + '\n')
            f.write('Hyperparameters (following Miracle team):\n')
            f.write(f'  - num_iter: {args.num_iter}\n')
            f.write(f'  - batch_size: {args.batch_size}\n')
            f.write(f'  - crop_size: {args.crop}\n')
            f.write(f'  - learning_rate: {args.learning_rate}\n')
            f.write(f'  - lr_warmup_iters: {args.lr_warmup_iters}\n')
            f.write(f'  - l1_weight: {args.l1_weight}\n')
            f.write(f'  - msssim_weight: {args.msssim_weight}\n')
            f.write(f'  - use_semantic_guidance: {args.use_semantic_guidance}\n')
            f.write('=' * 60 + '\n')
        
        # Initialize log files
        train_log_path = os.path.join(args.log_dir, 'train_log.csv')
        with open(train_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'total_loss', 'l1_loss', 'msssim_loss', 'learning_rate'])
        
        valid_log_path = os.path.join(args.log_dir, 'valid_log.csv')
        with open(valid_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'psnr', 'ssim', 'lpips', 'best_psnr', 'best_ssim', 'best_lpips'])
    
    # Setup loss functions
    msssim_loss_fn = MSSSIMLoss().to(device)
    
    # Setup dataloader
    train_dataset = STRRNetDataset(
        filename=f'data/train/{args.train_data}.txt',
        crop=args.crop,
        is_test=False,
        value_range=args.value_range
    )
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_worker > 0 else False
    )
    
    # Prepare with accelerator
    model, dataloader = accelerator.prepare(model, dataloader)
    
    # Setup optimizer (Adam as per STRRNet paper)
    # Use larger eps for fp16 stability
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-6  # Larger eps for numerical stability with fp16
    )
    
    # Setup LR scheduler (STRRNet specific)
    scheduler = get_strrnet_lr_scheduler(
        optimizer,
        num_iter=args.num_iter,
        warmup_iters=args.lr_warmup_iters
    )
    
    # Resume from checkpoint
    start_iter = 0
    best_psnr, best_ssim, best_lpips = -1, -1, float('inf')
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        accelerator.unwrap_model(model).load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_iter = checkpoint['iteration']
        best_psnr = checkpoint.get('best_psnr', -1)
        best_ssim = checkpoint.get('best_ssim', -1)
        best_lpips = checkpoint.get('best_lpips', float('inf'))
        print(f"Resumed from iteration {start_iter}")
    
    model.train()
    
    # Training loop
    pbar = tqdm(range(start_iter, args.num_iter), disable=not accelerator.is_main_process, desc="STRRNet Training")
    iter_loader = iter(dataloader)
    global_iter = start_iter
    
    # Loss accumulators
    iter_total_adder, iter_l1_adder, iter_msssim_adder, iter_count = 0, 0, 0, 0
    
    while global_iter < args.num_iter:
        batch = next(iter_loader, None)
        if batch is None:
            iter_loader = iter(dataloader)
            batch = next(iter_loader)
        
        input_img, label_img, semantic_labels = batch
        input_img = input_img.to(device, non_blocking=True)
        label_img = label_img.to(device, non_blocking=True)
        semantic_labels = semantic_labels.to(device, non_blocking=True)
        
        with accelerator.accumulate(model):
            with accelerator.autocast():
                # Forward pass with semantic labels
                if args.use_semantic_guidance:
                    pred_img = model(input_img, semantic_labels=semantic_labels)
                else:
                    pred_img = model(input_img, semantic_labels=None)
                
                # Handle multi-output models
                if isinstance(pred_img, (list, tuple)):
                    pred_img = pred_img[-1]
                
                # Compute losses with float32 for stability
                pred_float = pred_img.float()
                label_float = label_img.float()
                
                l1_loss = F.l1_loss(pred_float, label_float, reduction="mean")
                msssim_loss = msssim_loss_fn(pred_float, label_float)
                
                # Handle NaN in MS-SSIM
                if torch.isnan(msssim_loss):
                    msssim_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                # Total loss (STRRNet: L1 weight=1, MS-SSIM weight=0.2)
                total_loss = args.l1_weight * l1_loss + args.msssim_weight * msssim_loss
            
            # Backward
            accelerator.backward(total_loss)
            
            # Accumulate losses
            iter_total_adder += total_loss.item()
            iter_l1_adder += l1_loss.item()
            iter_msssim_adder += msssim_loss.item()
            iter_count += 1
            
            if accelerator.sync_gradients:
                # Gradient clipping
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                global_iter += 1
                
                if accelerator.is_main_process:
                    pbar.update(1)
                
                # Logging
                if (global_iter % args.print_freq) == 0:
                    avg_total = iter_total_adder / iter_count
                    avg_l1 = iter_l1_adder / iter_count
                    avg_msssim = iter_msssim_adder / iter_count
                    
                    # Reduce across processes
                    total_tensor = torch.tensor(avg_total).to(device)
                    l1_tensor = torch.tensor(avg_l1).to(device)
                    msssim_tensor = torch.tensor(avg_msssim).to(device)
                    
                    global_total = accelerator.reduce(total_tensor, reduction='mean').item()
                    global_l1 = accelerator.reduce(l1_tensor, reduction='mean').item()
                    global_msssim = accelerator.reduce(msssim_tensor, reduction='mean').item()
                    
                    if accelerator.is_main_process:
                        current_lr = optimizer.param_groups[0]["lr"]
                        
                        if use_wandb:
                            wandb.log({
                                "Train/Total": global_total,
                                "Train/L1": global_l1,
                                "Train/MS-SSIM": global_msssim,
                                "Learning Rate": current_lr
                            }, step=global_iter)
                        
                        # Log to CSV
                        train_log_path = os.path.join(args.log_dir, 'train_log.csv')
                        with open(train_log_path, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([global_iter, global_total, global_l1, global_msssim, current_lr])
                        
                        pbar.set_postfix({
                            'L1': f'{global_l1:.4f}',
                            'MS-SSIM': f'{global_msssim:.4f}',
                            'LR': f'{current_lr:.6f}'
                        })
                    
                    iter_total_adder, iter_l1_adder, iter_msssim_adder, iter_count = 0, 0, 0, 0
                
                # Validation and checkpoint
                if (global_iter % args.valid_freq) == 0:
                    accelerator.wait_for_everyone()
                    
                    if accelerator.is_main_process:
                        # Save model weights
                        save_name = os.path.join(args.model_save_dir, f'model_{global_iter}.pkl')
                        torch.save(accelerator.unwrap_model(model).state_dict(), save_name)
                        
                        # Save full checkpoint
                        checkpoint = {
                            'iteration': global_iter,
                            'model': accelerator.unwrap_model(model).state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'best_psnr': best_psnr,
                            'best_ssim': best_ssim,
                            'best_lpips': best_lpips,
                        }
                        ckpt_name = os.path.join(args.model_save_dir, 'checkpoint_latest.pt')
                        torch.save(checkpoint, ckpt_name)
                        
                        # Validation
                        psnr, ssim, lpips = _valid(accelerator.unwrap_model(model), accelerator, args)
                        
                        if psnr >= best_psnr:
                            best_psnr = psnr
                            torch.save(
                                accelerator.unwrap_model(model).state_dict(),
                                os.path.join(args.model_save_dir, 'best_psnr.pkl')
                            )
                        if ssim >= best_ssim:
                            best_ssim = ssim
                        if lpips <= best_lpips:
                            best_lpips = lpips
                            torch.save(
                                accelerator.unwrap_model(model).state_dict(),
                                os.path.join(args.model_save_dir, 'best_lpips.pkl')
                            )
                        
                        if use_wandb:
                            wandb.log({
                                "Valid/PSNR": psnr,
                                "Valid/SSIM": ssim,
                                "Valid/LPIPS": lpips,
                                "Valid/Best_PSNR": best_psnr,
                                "Valid/Best_SSIM": best_ssim,
                                "Valid/Best_LPIPS": best_lpips
                            }, step=global_iter)
                        
                        # Log to CSV
                        valid_log_path = os.path.join(args.log_dir, 'valid_log.csv')
                        with open(valid_log_path, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([global_iter, psnr, ssim, lpips, best_psnr, best_ssim, best_lpips])
                    
                    accelerator.wait_for_everyone()
    
    # Save final model
    if accelerator.is_main_process:
        save_name = os.path.join(args.model_save_dir, 'Final.pkl')
        torch.save(accelerator.unwrap_model(model).state_dict(), save_name)
        
        checkpoint = {
            'iteration': global_iter,
            'model': accelerator.unwrap_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_psnr': best_psnr,
            'best_ssim': best_ssim,
            'best_lpips': best_lpips,
        }
        ckpt_name = os.path.join(args.model_save_dir, 'checkpoint_final.pt')
        torch.save(checkpoint, ckpt_name)
        
        print(f"\nTraining completed!")
        print(f"Best PSNR: {best_psnr:.4f}")
        print(f"Best SSIM: {best_ssim:.4f}")
        print(f"Best LPIPS: {best_lpips:.4f}")
    
    pbar.close()
    
    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='STRRNet Training')
    
    # Model settings
    parser.add_argument('--model_name', type=str, default='STRRNet')
    parser.add_argument('--use_semantic_guidance', action='store_true', default=True,
                        help='Use semantic guidance module')
    parser.add_argument('--use_bg_subnet', action='store_true', default=True,
                        help='Use background restoration subnetwork')
    
    # Data settings
    parser.add_argument('--train_data', type=str, default='RaindropClarity')
    parser.add_argument('--test_data', type=str, default='RaindropClarity')
    parser.add_argument('--value_range', type=tuple, default=(0, 1))
    
    # Training settings (following STRRNet paper)
    parser.add_argument('--num_iter', type=int, default=500000,
                        help='Total training iterations (STRRNet: 500,000)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--crop', type=int, default=128,
                        help='Crop size (STRRNet: 128)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Initial learning rate (STRRNet: 0.0003)')
    parser.add_argument('--lr_warmup_iters', type=int, default=92000,
                        help='Constant LR iterations before decay (STRRNet: 92,000)')
    
    # Loss weights (STRRNet: L1=1, MS-SSIM=0.2)
    parser.add_argument('--l1_weight', type=float, default=1.0)
    parser.add_argument('--msssim_weight', type=float, default=0.2)
    
    # Other settings
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--valid_freq', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    
    # Experiment settings
    parser.add_argument('--exp_name', type=str, default='STRRNet_base')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                        choices=['no', 'fp16', 'bf16'])
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Initialize accelerator with DDP kwargs for unused parameters
    from accelerate import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=1,
        kwargs_handlers=[ddp_kwargs]
    )
    
    # Create model
    model = STRRNet(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        use_semantic_guidance=args.use_semantic_guidance,
        use_bg_subnet=args.use_bg_subnet
    )
    
    if accelerator.is_main_process:
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"STRRNet Parameters: {params:.2f}M")
        print(f"Use Semantic Guidance: {args.use_semantic_guidance}")
        print(f"Use Background Subnet: {args.use_bg_subnet}")
    
    # Train
    train_strrnet(model, accelerator, args)


if __name__ == '__main__':
    main()
