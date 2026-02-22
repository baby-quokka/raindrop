import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
from data import test_dataloader
import time
from utils import Adder, norm_range
import lpips as LPIPS
from torchvision.utils import save_image


def _tta_transform(x: torch.Tensor, rot: int, hflip: bool) -> torch.Tensor:
    """
    x: [B, C, H, W]
    rot: 0,1,2,3 -> 0,90,180,270 degree rotations
    hflip: horizontal flip 여부
    """
    if rot % 4 != 0:
        x = torch.rot90(x, k=rot, dims=(2, 3))
    if hflip:
        x = torch.flip(x, dims=(3,))
    return x


def _tta_inverse_transform(x: torch.Tensor, rot: int, hflip: bool) -> torch.Tensor:
    """
    _tta_transform의 역변환
    """
    if hflip:
        x = torch.flip(x, dims=(3,))
    if rot % 4 != 0:
        x = torch.rot90(x, k=4 - rot, dims=(2, 3))
    return x


@torch.no_grad()
def forward_x8_self_ensemble(model, inp: torch.Tensor):
    """
    x8 self-ensemble (rotation 0/90/180/270 x hflip on/off)
    inp: [B, C, H, W]
    반환: 평균 ensemble 출력 [B, C, H, W]
    """
    preds = []
    for rot in range(4):
        for hflip in (False, True):
            x_aug = _tta_transform(inp, rot, hflip)
            out = model(x_aug)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            out = _tta_inverse_transform(out, rot, hflip)
            preds.append(out)
    pred = torch.stack(preds, dim=0).mean(dim=0)
    return pred


# -----------------------------
# RGB -> Y (BT.601 digital, with offset) - pyiqa style
# -----------------------------
def rgb_to_y_bt601(img: torch.Tensor) -> torch.Tensor:
    """
    img: [B,3,H,W], range [0,1]
    return: [B,1,H,W], range approx [0,1]
    BT.601 digital Y' (studio swing) like many IQA/SR evals:
      Y = (16 + 65.481 R + 128.553 G + 24.966 B) / 255
    where R,G,B are in [0,1] scaled to [0,255] implicitly.
    """
    r = img[:, 0:1, :, :]
    g = img[:, 1:2, :, :]
    b = img[:, 2:3, :, :]
    y = (16.0/255.0) + (65.481/255.0)*r + (128.553/255.0)*g + (24.966/255.0)*b
    return y


def _crop_border(x: torch.Tensor, crop_border: int) -> torch.Tensor:
    if crop_border <= 0:
        return x
    return x[..., crop_border:-crop_border, crop_border:-crop_border]


# -----------------------------
# PSNR (pyiqa-like)
# -----------------------------
@torch.no_grad()
def psnr_y(
    pred: torch.Tensor,
    target: torch.Tensor,
    crop_border: int = 0,
    test_y_channel: bool = True,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    pred/target: [B,3,H,W] in [0,1]
    returns: scalar tensor (batch mean)
    """
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)

    if test_y_channel:
        pred = rgb_to_y_bt601(pred)
        target = rgb_to_y_bt601(target)

    pred = _crop_border(pred, crop_border)
    target = _crop_border(target, crop_border)

    # per-image MSE (mean over C,H,W)
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr.mean()


# -----------------------------
# SSIM (classic 11x11 Gaussian, sigma=1.5) - pyiqa style
# -----------------------------
def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device=None, dtype=None):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_1d = g.view(1, 1, 1, -1)  # [1,1,1,W]
    kernel_2d = (g[:, None] @ g[None, :]).view(1, 1, window_size, window_size)  # [1,1,H,W]
    return kernel_2d


def _filter2d(x: torch.Tensor, kernel: torch.Tensor):
    # depthwise conv
    c = x.shape[1]
    kernel = kernel.to(device=x.device, dtype=x.dtype)
    kernel = kernel.repeat(c, 1, 1, 1)  # [C,1,Kh,Kw]
    return F.conv2d(x, kernel, padding=kernel.shape[-1] // 2, groups=c)


@torch.no_grad()
def ssim_y(
    pred: torch.Tensor,
    target: torch.Tensor,
    crop_border: int = 0,
    test_y_channel: bool = True,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """
    pred/target: [B,3,H,W] in [0,1]
    returns: scalar tensor (batch mean)
    """
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)

    if test_y_channel:
        pred = rgb_to_y_bt601(pred)
        target = rgb_to_y_bt601(target)

    pred = _crop_border(pred, crop_border)
    target = _crop_border(target, crop_border)

    # constants for data_range=1.0
    c1 = (k1 * 1.0) ** 2
    c2 = (k2 * 1.0) ** 2

    kernel = _gaussian_kernel(window_size, sigma, device=pred.device, dtype=pred.dtype)

    mu1 = _filter2d(pred, kernel)
    mu2 = _filter2d(target, kernel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = _filter2d(pred * pred, kernel) - mu1_sq
    sigma2_sq = _filter2d(target * target, kernel) - mu2_sq
    sigma12   = _filter2d(pred * target, kernel) - mu12

    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    # mean over C,H,W per image then batch mean
    return ssim_map.mean(dim=(1, 2, 3)).mean()


def _eval(model, args):
    loaded = torch.load(args.test_model, map_location=lambda storage, loc: storage)
    state_dict = loaded.get('model', loaded) if isinstance(loaded, dict) and 'model' in loaded else loaded
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # test_txt가 지정되면 해당 경로 사용, 아니면 기본 경로
    test_txt_path = args.test_txt if hasattr(args, 'test_txt') and args.test_txt else f'data/test/{args.test_data}.txt'

    # valid 경로(data/valid/...)는 챌린지 제출용: Drop만 있고 GT(Clear)가 없음.
    # → paired=False 로 불러오고, PSNR/SSIM/LPIPS 계산은 건너뛴다.
    if 'data/valid/' in test_txt_path or '/valid/' in test_txt_path:
        has_gt = False
        dataloader = test_dataloader(test_txt_path, args.value_range, paired=False, batch_size=1, num_workers=1)
    else:
        has_gt = True
        dataloader = test_dataloader(test_txt_path, args.value_range, paired=True, batch_size=1, num_workers=1)
    torch.cuda.empty_cache()
    
    # Initialize LPIPS metric
    lpips_fn = LPIPS.LPIPS(net='alex').to(device)
    lpips_fn.eval()
    
    adder = Adder()
    model.eval(), model.to(device)
    factor = 32

    # 결과 저장 경로: 외부에서 지정된 result_dir이 있으면 사용, 아니면 기본 경로
    if hasattr(args, 'result_dir') and getattr(args, 'result_dir'):
        result_dir = args.result_dir
    else:
        result_dir = os.path.join('results', f"{args.exp_name}", f'{args.test_data}')
    os.makedirs(result_dir, exist_ok=True)
    args.result_dir = result_dir
    
    # Uformer는 정사각형 이미지만 지원
    is_uformer = model.__class__.__name__ == 'Uformer'
    
    # 개별 이미지 결과를 저장할 리스트
    results_per_image = []
    
    with torch.no_grad():
        if has_gt:
            psnr_adder = Adder()
            ssim_adder = Adder()
            lpips_adder = Adder()
        else:
            psnr_adder = None
            ssim_adder = None
            lpips_adder = None

        for iter_idx, data in enumerate(dataloader):
            if has_gt:
                input_img, label_img, name = data
                input_img, label_img = input_img.to(device), label_img.to(device)
            else:
                input_img, name = data
                input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            
            if is_uformer:
                # Uformer: 정사각형으로 패딩
                max_size = max(h, w)
                H = W = ((max_size + factor) // factor) * factor
            else:
                H = ((h + factor) // factor) * factor
                W = ((w + factor) // factor) * factor
            
            padh = H - h
            padw = W - w
            input_img = F.pad(input_img, (0, padw, 0, padh), 'reflect')
           
            tm = time.time()

            # x8 self-ensemble 사용 여부
            use_self_ensemble = getattr(args, 'use_self_ensemble', False)
            if use_self_ensemble:
                pred = forward_x8_self_ensemble(model, input_img)
            else:
                pred = model(input_img)
                # 모델이 multi-scale 리스트를 반환하는 경우 마지막 출력(원본 스케일) 사용
                if isinstance(pred, (list, tuple)):
                    pred = pred[-1]
            pred = pred[:,:,:h,:w]
            pred = norm_range(pred, value_range=(0, 1))

            elapsed = time.time() - tm
            adder(elapsed)

            if has_gt:
                # Calculate metrics (pyiqa-style)
                psnr = psnr_y(pred, label_img).item()
                psnr_adder(psnr)
                
                ssim_val = ssim_y(pred, label_img).item()
                ssim_adder(ssim_val)
                
                # LPIPS expects input in range [-1, 1]
                lpips_val = lpips_fn(pred * 2 - 1, label_img * 2 - 1).item()
                lpips_adder(lpips_val)
            else:
                psnr = None
                ssim_val = None
                lpips_val = None
            
            # 개별 결과 저장
            results_per_image.append({
                'image_name': name[0],
                'psnr': psnr if psnr is not None else 'N/A',
                'ssim': ssim_val if ssim_val is not None else 'N/A',
                'lpips': lpips_val if lpips_val is not None else 'N/A',
                'time': elapsed
            })

            if has_gt:
                print(f'{name[0]} | PSNR: {psnr:.2f} ssim: {ssim_val:.5f} lpips: {lpips_val:.5f}')
            else:
                print(f'{name[0]} | time: {elapsed:.4f}s')
            # max_save_images: 너무 많은 이미지를 저장하지 않기 위한 상한
            max_save = getattr(args, 'max_save_images', -1)
            if args.save_image and (max_save < 0 or iter_idx < max_save):
                base_name = name[0]
                # submission/valid 셋(=GT 없음)일 때는 입력 파일명 그대로 저장 (예: 00001.png)
                if not has_gt:
                    # 데이터셋 name은 확장자 없을 수 있음 → .png 보장
                    if not base_name or not base_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        base_name = (base_name or 'out') + '.png'
                    save_name = os.path.join(args.result_dir, base_name)
                else:
                    # GT가 있는 일반 평가 셋에서는 이름 충돌 방지를 위해 인덱스를 prefix로 붙임
                    save_name = os.path.join(args.result_dir, f'{iter_idx:04d}_{base_name}.png')
                save_image(pred, save_name, normalize=True, value_range=args.value_range)

        # 평균 결과
        avg_time = adder.average()
        if has_gt:
            avg_psnr = psnr_adder.average()
            avg_ssim = ssim_adder.average()
            avg_lpips = lpips_adder.average()
        else:
            avg_psnr = None
            avg_ssim = None
            avg_lpips = None
        
        print('==========================================================')
        print(f'Evaluation done for {args.test_data}')
        if has_gt:
            print('The average PSNR is %.2f dB' % avg_psnr)
            print('The average SSIM is %.5f' % avg_ssim)
            print('The average LPIPS is %.5f' % avg_lpips)
        else:
            print('No GT available (submission/valid set) → PSNR/SSIM/LPIPS not computed.')
        print("Average time: %f" % avg_time)
        
        # ═══════════════════════════════════════════════════════════════
        # 결과를 CSV 파일로 저장
        # ═══════════════════════════════════════════════════════════════
        
        # 1. 개별 이미지 결과 CSV
        csv_per_image_path = os.path.join(args.result_dir, 'results_per_image.csv')
        with open(csv_per_image_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image_name', 'psnr', 'ssim', 'lpips', 'time'])
            writer.writeheader()
            writer.writerows(results_per_image)
        print(f'Per-image results saved to: {csv_per_image_path}')
        
        # 2. 요약 결과 CSV (평균값)
        csv_summary_path = os.path.join(args.result_dir, 'results_summary.csv')
        with open(csv_summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['exp_name', args.exp_name])
            writer.writerow(['test_data', args.test_data])
            if has_gt:
                writer.writerow(['avg_psnr', f'{avg_psnr:.4f}'])
                writer.writerow(['avg_ssim', f'{avg_ssim:.6f}'])
                writer.writerow(['avg_lpips', f'{avg_lpips:.6f}'])
            else:
                writer.writerow(['avg_psnr', 'N/A'])
                writer.writerow(['avg_ssim', 'N/A'])
                writer.writerow(['avg_lpips', 'N/A'])
            writer.writerow(['avg_time', f'{avg_time:.6f}'])
            writer.writerow(['num_images', len(results_per_image)])
        print(f'Summary results saved to: {csv_summary_path}')
        
        # 3. 간단한 텍스트 파일 (한눈에 보기 좋게)
        txt_name = getattr(args, 'summary_txt_name', 'results.txt')
        txt_path = os.path.join(args.result_dir, txt_name)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f'Experiment: {args.exp_name}\n')
            f.write(f'Test Data: {args.test_data}\n')
            f.write(f'Number of Images: {len(results_per_image)}\n')
            f.write('='*50 + '\n')
            if has_gt:
                f.write(f'Average PSNR:  {avg_psnr:.2f} dB\n')
                f.write(f'Average SSIM:  {avg_ssim:.5f}\n')
                f.write(f'Average LPIPS: {avg_lpips:.5f}\n')
            else:
                f.write('Average PSNR:  N/A (no GT, submission/valid set)\n')
                f.write('Average SSIM:  N/A (no GT, submission/valid set)\n')
                f.write('Average LPIPS: N/A (no GT, submission/valid set)\n')
            f.write(f'Average Time:  {avg_time:.4f} sec\n')
            f.write('='*50 + '\n')
        print(f'Text results saved to: {txt_path}')

