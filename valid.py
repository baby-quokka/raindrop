import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from data import valid_dataloader
from utils import Adder, norm_range
from eval import psnr_y, ssim_y
import lpips as LPIPS

def _valid(model, accelerator, args):
    device = accelerator.device
    dataset = valid_dataloader(f'data/test/{args.test_data}.txt', value_range=args.value_range, paired=True, batch_size=1, num_workers=1)
    
    # Initialize LPIPS metric
    lpips_fn = LPIPS.LPIPS(net='alex').to(device)
    lpips_fn.eval()
    
    model.eval()
    psnr_adder = Adder()
    ssim_adder = Adder()
    lpips_adder = Adder()

    # Uformer는 정사각형 이미지만 지원
    is_uformer = model.__class__.__name__ == 'Uformer'
    
    with torch.no_grad():
        print('Start Evaluation')
        factor = 32
        for idx, data in enumerate(dataset):
            input_img, label_img, name = data
            input_img, label_img = input_img.to(device), label_img.to(device)

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

            # STRRNet의 경우 semantic label 생성 (inference 모드)
            if hasattr(model, 'use_semantic_guidance') and model.use_semantic_guidance:
                # Inference에서는 semantic label을 None으로 전달 (모델 내부에서 자동 분류)
                pred = model(input_img, semantic_labels=None)
            else:
                pred = model(input_img)
            
            # 모델이 multi-scale 리스트를 반환하는 경우 마지막 출력(원본 스케일) 사용
            if isinstance(pred, (list, tuple)):
                pred = pred[-1]
            pred = pred[:,:,:h,:w]

            pred = norm_range(pred, value_range=(0, 1))

            # Calculate metrics (pyiqa-style)
            psnr = psnr_y(pred, label_img).item()
            psnr_adder(psnr)
            
            ssim_val = ssim_y(pred, label_img).item()
            ssim_adder(ssim_val)
            
            # LPIPS expects input in range [-1, 1]
            lpips_val = lpips_fn(pred * 2 - 1, label_img * 2 - 1).item()
            lpips_adder(lpips_val)

            print('\r%03d'%idx, end=' ')

    print('\n')
    model.train()
    return psnr_adder.average(), ssim_adder.average(), lpips_adder.average()
