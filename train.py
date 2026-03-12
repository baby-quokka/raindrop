import os
import csv
import json
import types
import copy
import torch
from data import train_dataloader
from valid import _valid
import torch.nn.functional as F
from tqdm import tqdm
import lpips as LPIPS

# 일부 diffusers 버전은 torch.xpu(empty_cache)를 참조하는데,
# 현재 PyTorch 빌드에는 torch.xpu 속성이 없어 AttributeError가 발생할 수 있음.
# XPU를 쓰지 않으므로, 누락된 경우 dummy 객체를 주입해 안전하게 우회한다.
if not hasattr(torch, "xpu"):
    torch.xpu = types.SimpleNamespace(empty_cache=lambda: None)

from diffusers.optimization import get_scheduler


# ============================
# 추가 Loss 함수들 (MS-SSIM, FFT)
# ============================
try:
    # pip install pytorch-msssim 필요
    from pytorch_msssim import ms_ssim as ms_ssim_fn
except ImportError:
    ms_ssim_fn = None


def compute_ms_ssim_loss(pred, target, data_range=1.0):
    """
    MS-SSIM 기반 loss: 1 - MS-SSIM
    pred, target: [B, C, H, W], 값 범위 [0, data_range]
    """
    if ms_ssim_fn is None:
        # 라이브러리가 없으면 0으로 둬서 학습은 진행되도록 함
        return pred.new_tensor(0.0)

    # data_range가 (min, max) 형태의 튜플이면 실제 범위 길이로 변환
    if isinstance(data_range, (tuple, list)):
        if len(data_range) == 2:
            lo, hi = float(data_range[0]), float(data_range[1])
            data_range_val = hi - lo
            clamp_min, clamp_max = lo, hi
        else:
            data_range_val = 1.0
            clamp_min, clamp_max = 0.0, 1.0
    else:
        data_range_val = float(data_range)
        clamp_min, clamp_max = 0.0, data_range_val

    # 혼합 정밀도(fp16)에서 수치 안정성을 위해 float32 + clamp 사용
    pred_f = pred.float().clamp(clamp_min, clamp_max)
    target_f = target.float().clamp(clamp_min, clamp_max)

    # size_average=True 로 배치 평균
    ms_ssim_val = ms_ssim_fn(pred_f, target_f, data_range=data_range_val, size_average=True)
    loss = 1.0 - ms_ssim_val

    # NaN/Inf 방지: 이상치면 0으로
    if not torch.isfinite(loss):
        return pred.new_tensor(0.0)
    return loss


def compute_fft_loss(pred, target):
    """
    간단한 FFT L1 loss:
    - 주파수 도메인 magnitude 스펙트럼 사이의 L1 distance
    """
    # 혼합 정밀도(fp16)에서 수치 폭발을 막기 위해 float32에서 계산
    pred_f = pred.float()
    target_f = target.float()

    # 2D FFT (real-to-complex), 공간 차원만 변환
    pred_fft = torch.fft.rfftn(pred_f, dim=(2, 3))
    target_fft = torch.fft.rfftn(target_f, dim=(2, 3))

    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)

    loss = F.l1_loss(pred_mag, target_mag, reduction="mean")

    # NaN/Inf 방지
    if not torch.isfinite(loss):
        return pred.new_tensor(0.0)
    return loss


def _train(model, accelerator, args):
    device = accelerator.device
    use_wandb = getattr(args, 'use_wandb', False)
    
    # WandB import (조건부)
    if use_wandb:
        import wandb
    
    if accelerator.is_main_process:
        if use_wandb:
            wandb.init(project="NTIRE2026", name=f"{args.exp_name}", config=vars(args))
        
        args.model_save_dir = os.path.join('results', f"{args.exp_name}", 'ckpt')
        args.log_dir = os.path.join('results', f"{args.exp_name}", 'logs')
        os.makedirs(args.model_save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        
        # ═══════════════════════════════════════════════════════════════
        # 하이퍼파라미터 & 설정 저장 (JSON)
        # ═══════════════════════════════════════════════════════════════
        config_path = os.path.join(args.log_dir, 'config.json')
        config_dict = {}
        for key, value in vars(args).items():
            # tuple은 JSON 호환을 위해 list로 변환
            if isinstance(value, tuple):
                config_dict[key] = list(value)
            else:
                config_dict[key] = value
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        print(f'Config saved to: {config_path}')
        
        # 읽기 쉬운 텍스트 버전도 저장
        config_txt_path = os.path.join(args.log_dir, 'config.txt')
        with open(config_txt_path, 'w', encoding='utf-8') as f:
            f.write(f'Experiment: {args.exp_name}\n')
            f.write('='*60 + '\n')
            f.write(f'Model: {args.model_name}\n')
            f.write(f'Train Data: {args.train_data}\n')
            f.write(f'Test Data: {args.test_data}\n')
            f.write('='*60 + '\n')
            f.write('Hyperparameters:\n')
            f.write(f'  - num_iter: {args.num_iter}\n')
            f.write(f'  - batch_size: {args.batch_size}\n')
            f.write(f'  - crop_size: {args.crop}\n')
            f.write(f'  - learning_rate: {args.learning_rate}\n')
            f.write(f'  - lr_scheduler: {args.lr_scheduler}\n')
            f.write(f'  - lr_warmup_steps: {args.lr_warmup_steps}\n')
            f.write(f'  - mixed_precision: {args.mixed_precision}\n')
            f.write(f'  - grad_accum: {args.grad_accum}\n')  # 배치 사이즈와 곱 잘 계산해서 multi
            f.write(f'  - max_grad_norm: {args.max_grad_norm}\n')
            f.write(f'  - use_lpips: {args.use_lpips}\n')
            f.write(f'  - lambda_l1: {getattr(args, "lambda_l1", 1.0)}\n')
            f.write(f'  - lambda_lpips: {args.lambda_lpips}\n')
            f.write(f'  - lambda_msssim: {getattr(args, "lambda_msssim", 0.1)}\n')
            f.write(f'  - lambda_fft: {getattr(args, "lambda_fft", 0.01)}\n')
            f.write(f'  - use_ema: {getattr(args, "use_ema", False)}\n')
            f.write(f'  - ema_decay: {getattr(args, "ema_decay", 0.999)}\n')
            f.write(f'  - value_range: {args.value_range}\n')
            f.write(f'  - seed: {args.seed}\n')
            f.write('='*60 + '\n')
        print(f'Config (text) saved to: {config_txt_path}')
        
        # 학습 로그 CSV 파일 초기화
        train_log_path = os.path.join(args.log_dir, 'train_log.csv')
        with open(train_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'l1_loss', 'lpips_loss', 'msssim_loss', 'fft_loss', 'total_loss', 'learning_rate'])
        
        # 검증 로그 CSV 파일 초기화
        valid_log_path = os.path.join(args.log_dir, 'valid_log.csv')
        with open(valid_log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'psnr', 'ssim', 'lpips', 'best_psnr', 'best_ssim', 'best_lpips'])

    if args.use_lpips:
        lpips_fn = LPIPS.LPIPS(net='vgg').to(device)
        lpips_fn.requires_grad_(False), lpips_fn.eval()
    dataloader = train_dataloader(f'data/train/{args.train_data}.txt', args.value_range, args.crop, args.batch_size, args.num_worker)

    model, dataloader = accelerator.prepare(model, dataloader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps, 
        num_training_steps=args.num_iter,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power
    )

    # EMA (Exponential Moving Average): valid/save 시 EMA 가중치 사용
    use_ema = getattr(args, 'use_ema', False)
    ema_decay = getattr(args, 'ema_decay', 0.999)
    ema_state = None
    if use_ema:
        ema_state = copy.deepcopy(accelerator.unwrap_model(model).state_dict())
        if accelerator.is_main_process:
            print(f"EMA enabled (decay={ema_decay})")
    
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
        if use_ema and 'ema_state' in checkpoint:
            ema_state = {k: v.to(accelerator.device) for k, v in checkpoint['ema_state'].items()}
            print("Resumed EMA state")
        print(f"Resumed from iteration {start_iter}")
    
    model.train()

    pbar = tqdm(range(start_iter, args.num_iter), disable=not accelerator.is_main_process, desc="Training")
    iter_loader = iter(dataloader)
    global_iter = start_iter
    iter_l1_adder = iter_lpips_adder = 0.0
    iter_msssim_adder = iter_fft_adder = 0.0
    iter_loss_adder = 0.0
    iter_adder = 0

    while global_iter < args.num_iter:
        batch = next(iter_loader, None)
        if batch is None:
            iter_loader = iter(dataloader)
            batch = next(iter_loader, None)

        input_img, label_img = batch
        input_img = input_img.to(device, non_blocking=True)
        label_img = label_img.to(device, non_blocking=True)

        with accelerator.accumulate(model):
            with accelerator.autocast():
                pred_img = model(input_img)
                
                # 모델이 multi-scale 리스트를 반환하는 경우 처리
                if isinstance(pred_img, (list, tuple)):
                    # Multi-scale loss를 위해 리스트 유지 (나중에 처리)
                    pred_img_list = pred_img
                    pred_img = pred_img[-1]  # 메인 출력은 마지막 스케일
                else:
                    pred_img_list = None
            
            # Multi-scale loss 계산 (ConvIR 스타일)
            if pred_img_list is not None and len(pred_img_list) > 1:
                # 각 스케일에서 loss 계산
                scale_weights = [0.25, 0.5, 1.0]  # x/4, x/2, x 스케일 가중치
                l1_losses = []
                lpips_losses = []
                
                for i, pred_scale in enumerate(pred_img_list):
                    # Label을 해당 스케일로 리사이즈
                    if i == 0:  # x/4
                        label_scale = F.interpolate(label_img, size=pred_scale.shape[2:], mode='bilinear', align_corners=False)
                    elif i == 1:  # x/2
                        label_scale = F.interpolate(label_img, size=pred_scale.shape[2:], mode='bilinear', align_corners=False)
                    else:  # x (원본)
                        label_scale = label_img
                    
                    l1_scale = F.l1_loss(pred_scale, label_scale, reduction="mean")
                    l1_losses.append(l1_scale * scale_weights[i])
                    
                    if args.use_lpips:
                        lpips_scale = lpips_fn(pred_scale * 2 - 1, label_scale * 2 - 1).mean()
                        lpips_losses.append(lpips_scale * scale_weights[i])
                
                # 가중 평균
                l1_loss = sum(l1_losses) / sum(scale_weights)
                if args.use_lpips:
                    lpips_loss = sum(lpips_losses) / sum(scale_weights)
                else:
                    lpips_loss = torch.tensor(0.0, device=device)
            else:
                # 단일 스케일 loss
                l1_loss = F.l1_loss(pred_img, label_img, reduction="mean")
                if args.use_lpips:
                    lpips_loss = lpips_fn(pred_img * 2 - 1, label_img * 2 - 1).mean()
                else:
                    lpips_loss = torch.tensor(0.0, device=device)

            # ============================
            # 추가 Loss: MS-SSIM, FFT
            # ============================
            # args에 값이 없으면 기본값 사용 (로그 스케일에 맞춰 FFT는 0.01 권장)
            lambda_msssim = getattr(args, "lambda_msssim", 0.1)
            lambda_fft = getattr(args, "lambda_fft", 0.01)

            # 값 범위는 [0, 1]이라고 가정 (args.value_range를 쓰는 경우 필요시 조정)
            data_range = getattr(args, "value_range", 1.0)

            msssim_loss = compute_ms_ssim_loss(pred_img, label_img, data_range=data_range)
            fft_loss = compute_fft_loss(pred_img, label_img)

            lambda_l1 = getattr(args, "lambda_l1", 1.0)
            loss = (
                lambda_l1 * l1_loss
                + args.lambda_lpips * lpips_loss
                + lambda_msssim * msssim_loss
                + lambda_fft * fft_loss
            )

            # loss가 NaN/Inf이면 해당 스텝은 스킵 (가중치 오염 방지)
            # 단, iter/scheduler는 진행시켜서 같은 iter에서 무한 반복되지 않도록 함
            if not torch.isfinite(loss):
                if accelerator.is_main_process:
                    try:
                        l1_v, lp_v = l1_loss.item(), lpips_loss.item()
                    except Exception:
                        l1_v, lp_v = float('nan'), float('nan')
                    print(f"[Warning] Non-finite loss at iter {global_iter}: "
                          f"L1={l1_v}, LPIPS={lp_v}, MS-SSIM={msssim_loss.item():.4f}, FFT={fft_loss.item():.4f} (skipping update, advancing step)")
                optimizer.zero_grad(set_to_none=True)
                # NaN이어도 iter/scheduler는 무조건 진행시켜서 같은 iter에서 무한 반복 방지
                scheduler.step()
                global_iter += 1
                if accelerator.is_main_process:
                    pbar.update(1)
                if (global_iter % args.valid_freq) == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        net = accelerator.unwrap_model(model)
                        # 검증·저장 시 EMA 가중치 사용
                        if use_ema and ema_state is not None:
                            cur_state = copy.deepcopy(net.state_dict())
                            net.load_state_dict(ema_state, strict=True)
                        save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % global_iter)
                        torch.save(net.state_dict(), save_name)
                        checkpoint = {
                            'iteration': global_iter,
                            'model': accelerator.unwrap_model(model).state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'best_psnr': best_psnr,
                            'best_ssim': best_ssim,
                            'best_lpips': best_lpips,
                        }
                        if use_ema and ema_state is not None:
                            checkpoint['ema_state'] = {k: v.cpu() for k, v in ema_state.items()}
                        ckpt_name = os.path.join(args.model_save_dir, 'checkpoint_latest.pt')
                        torch.save(checkpoint, ckpt_name)
                        psnr, ssim, lpips_val = _valid(net, accelerator, args)
                        if use_ema and ema_state is not None:
                            net.load_state_dict(cur_state, strict=True)
                        if psnr >= best_psnr: best_psnr = psnr
                        if ssim >= best_ssim: best_ssim = ssim
                        if lpips_val <= best_lpips: best_lpips = lpips_val
                        if use_wandb:
                            wandb.log({"Valid/PSNR": psnr, "Valid/SSIM": ssim, "Valid/LPIPS": lpips_val}, step=global_iter)
                            wandb.log({"Valid/Best PSNR": best_psnr, "Valid/Best SSIM": best_ssim, "Valid/Best LPIPS": best_lpips}, step=global_iter)
                        valid_log_path = os.path.join(args.log_dir, 'valid_log.csv')
                        with open(valid_log_path, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([global_iter, psnr, ssim, lpips_val, best_psnr, best_ssim, best_lpips])
                    accelerator.wait_for_everyone()
                continue

            accelerator.backward(loss)
            iter_l1_adder += float(l1_loss.item())
            iter_lpips_adder += float(lpips_loss.item())
            iter_msssim_adder += float(msssim_loss.item())
            iter_fft_adder += float(fft_loss.item())
            iter_loss_adder += float(loss.item())
            iter_adder += 1

            if accelerator.sync_gradients:    
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # EMA 업데이트 (학습 가중치의 이동 평균)
                if use_ema and ema_state is not None:
                    with torch.no_grad():
                        unwrapped = accelerator.unwrap_model(model)
                        for k, v in unwrapped.state_dict().items():
                            if v.dtype in (torch.float32, torch.float16):
                                ema_state[k].mul_(ema_decay).add_(v.to(ema_state[k].dtype), alpha=1.0 - ema_decay)
                            else:
                                ema_state[k].copy_(v)
                                
                scheduler.step()
                global_iter += 1
                if accelerator.is_main_process:
                    pbar.update(1)
                if (global_iter % args.print_freq) == 0 and iter_adder > 0:
                    local_l1_loss = iter_l1_adder / iter_adder
                    local_lpips_loss = iter_lpips_adder / iter_adder
                    local_msssim_loss = iter_msssim_adder / iter_adder
                    local_fft_loss = iter_fft_adder / iter_adder
                    local_total_loss = iter_loss_adder / iter_adder

                    l1_loss_tensor = torch.tensor(local_l1_loss, device=device)
                    lpips_loss_tensor = torch.tensor(local_lpips_loss, device=device)
                    msssim_loss_tensor = torch.tensor(local_msssim_loss, device=device)
                    fft_loss_tensor = torch.tensor(local_fft_loss, device=device)
                    total_loss_tensor = torch.tensor(local_total_loss, device=device)

                    global_l1_loss = accelerator.reduce(l1_loss_tensor, reduction='mean').item()
                    global_lpips_loss = accelerator.reduce(lpips_loss_tensor, reduction='mean').item()
                    global_msssim_loss = accelerator.reduce(msssim_loss_tensor, reduction='mean').item()
                    global_fft_loss = accelerator.reduce(fft_loss_tensor, reduction='mean').item()
                    global_total_loss = accelerator.reduce(total_loss_tensor, reduction='mean').item()

                    if accelerator.is_main_process:
                        if use_wandb:
                            wandb.log({
                                "Train/L1": global_l1_loss,
                                "Train/LPIPS": global_lpips_loss,
                                "Train/MS-SSIM": global_msssim_loss,
                                "Train/FFT": global_fft_loss,
                                "Train/Total": global_total_loss,
                                "Learning Rate": optimizer.param_groups[0]["lr"],
                            }, step=global_iter)
                        # 학습 로그를 CSV에 저장
                        train_log_path = os.path.join(args.log_dir, 'train_log.csv')
                        with open(train_log_path, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                global_iter,
                                global_l1_loss,
                                global_lpips_loss,
                                global_msssim_loss,
                                global_fft_loss,
                                global_total_loss,
                                optimizer.param_groups[0]["lr"],
                            ])

                    iter_l1_adder = iter_lpips_adder = 0.0
                    iter_msssim_adder = iter_fft_adder = 0.0
                    iter_loss_adder = 0.0
                    iter_adder = 0
                if (global_iter % args.valid_freq) == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        net = accelerator.unwrap_model(model)
                        if use_ema and ema_state is not None:
                            cur_state = copy.deepcopy(net.state_dict())
                            net.load_state_dict(ema_state, strict=True)
                        # 모델 가중치만 저장 (평가/추론용 → EMA 사용 시 EMA 가중치)
                        save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % global_iter)
                        torch.save(net.state_dict(), save_name)
                        # 전체 checkpoint 저장 (resume용, model=학습 가중치)
                        checkpoint = {
                            'iteration': global_iter,
                            'model': accelerator.unwrap_model(model).state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'best_psnr': best_psnr,
                            'best_ssim': best_ssim,
                            'best_lpips': best_lpips,
                        }
                        if use_ema and ema_state is not None:
                            checkpoint['ema_state'] = {k: v.cpu() for k, v in ema_state.items()}
                        ckpt_name = os.path.join(args.model_save_dir, 'checkpoint_latest.pt')
                        torch.save(checkpoint, ckpt_name)
                        psnr, ssim, lpips_val = _valid(net, accelerator, args)
                        if use_ema and ema_state is not None:
                            net.load_state_dict(cur_state, strict=True)
                        if psnr >= best_psnr: best_psnr = psnr
                        if ssim >= best_ssim: best_ssim = ssim
                        if lpips_val <= best_lpips: best_lpips = lpips_val
                        if use_wandb:
                            wandb.log({"Valid/PSNR": psnr, "Valid/SSIM": ssim, "Valid/LPIPS": lpips_val}, step=global_iter)
                            wandb.log({"Valid/Best PSNR": best_psnr, "Valid/Best SSIM": best_ssim, "Valid/Best LPIPS": best_lpips}, step=global_iter)
                        # 검증 로그를 CSV에 저장
                        valid_log_path = os.path.join(args.log_dir, 'valid_log.csv')
                        with open(valid_log_path, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([global_iter, psnr, ssim, lpips_val, best_psnr, best_ssim, best_lpips])
                    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        net = accelerator.unwrap_model(model)
        train_state = net.state_dict()
        if use_ema and ema_state is not None:
            net.load_state_dict(ema_state, strict=True)
        # 최종 모델 가중치 저장 (추론용 → EMA 사용 시 EMA 가중치)
        save_name = os.path.join(args.model_save_dir, 'Final.pkl')
        torch.save(net.state_dict(), save_name)
        if use_ema and ema_state is not None:
            net.load_state_dict(train_state, strict=True)
        # 최종 checkpoint 저장 (resume용, model=학습 가중치)
        checkpoint = {
            'iteration': global_iter,
            'model': train_state,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_psnr': best_psnr,
            'best_ssim': best_ssim,
            'best_lpips': best_lpips,
        }
        if use_ema and ema_state is not None:
            checkpoint['ema_state'] = {k: v.cpu() for k, v in ema_state.items()}
        ckpt_name = os.path.join(args.model_save_dir, 'checkpoint_final.pt')
        torch.save(checkpoint, ckpt_name)

    pbar.close()
    if use_wandb:
        wandb.finish()
    
    # end_training()은 trackers가 있을 때만 호출
    if hasattr(accelerator, 'trackers') and accelerator.trackers:
        accelerator.end_training()