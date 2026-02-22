"""
Pseudo-GT를 사용하는 파인튜닝용 데이터셋

입력: Drop 이미지
타겟: Pseudo-GT (median fusion으로 생성된 이미지)

STRRNet 권장: geometric augmentation + mixup 비활성화 시 pseudo-GT 노이즈 학습 완화.
  → use_augment=False 로 center crop만 사용, flip 없음.
"""

import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from data.utils import *


class PseudoGTDataset(Dataset):
    """
    Pseudo-GT를 타겟으로 사용하는 데이터셋
    
    Args:
        input_list: 입력(Drop) 이미지 경로 리스트
        pseudo_gt_dir: Pseudo-GT 이미지가 저장된 디렉토리
        crop: 학습 시 crop 크기
        value_range: 값 범위 (0, 1) 또는 (-1, 1)
        use_augment: True면 random crop + flip, False면 center crop만 (STRRNet 스타일 권장)
    """
    def __init__(
        self,
        input_list: str,
        pseudo_gt_dir: str,
        crop: int = 256,
        value_range: Tuple[float, float] = (0, 1),
        use_augment: bool = True,
    ):
        super().__init__()
        self.pseudo_gt_dir = pseudo_gt_dir
        self.crop = crop
        self.value_range = value_range
        self.use_augment = use_augment
        
        # 입력 이미지 경로 리스트 읽기
        with open(input_list, "r") as f:
            self.input_files: List[str] = [ln.strip() for ln in f if ln.strip()]
        
        self.files = []
        for input_path in self.input_files:
            # Pseudo-GT 경로 생성 (파일명 기반)
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            pseudo_gt_path = os.path.join(pseudo_gt_dir, f"{base_name}.png")
            
            # Pseudo-GT 파일이 존재하는지 확인
            if os.path.exists(pseudo_gt_path):
                self.files.append((input_path, pseudo_gt_path))
            else:
                print(f"[Warning] Pseudo-GT not found: {pseudo_gt_path}, skipping")
        
        print(f"[Info] Loaded {len(self.files)} pairs from {len(self.input_files)} input images")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        input_path, pseudo_gt_path = self.files[idx]
        
        input_img = read_rgb(input_path)  # uint8 [3, H, W]
        pseudo_gt = read_rgb(pseudo_gt_path)  # uint8 [3, H, W]
        
        if self.use_augment:
            input_img, pseudo_gt = crop_pair(input_img, pseudo_gt, self.crop)
            input_img, pseudo_gt = flip_pair(input_img, pseudo_gt, p=0.5)
        else:
            input_img, pseudo_gt = center_crop_pair(input_img, pseudo_gt, self.crop)
        
        # 값 범위 변환
        if self.value_range == (0, 1):
            return to_m01(input_img), to_m01(pseudo_gt)
        elif self.value_range == (-1, 1):
            return to_m11(input_img), to_m11(pseudo_gt)
        else:
            raise ValueError(f"Unsupported value_range {self.value_range}")


def pseudo_gt_train_dataloader(
    input_list: str,
    pseudo_gt_dir: str,
    crop: int = 256,
    value_range: Tuple[float, float] = (0, 1),
    batch_size: int = 64,
    num_workers: int = 4,
    use_augment: bool = True,
):
    """
    Pseudo-GT 데이터셋용 학습 데이터로더.
    use_augment=False: STRRNet 권장 (geometric aug 비활성화).
    """
    dataset = PseudoGTDataset(
        input_list=input_list,
        pseudo_gt_dir=pseudo_gt_dir,
        crop=crop,
        value_range=value_range,
        use_augment=use_augment,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    
    return dataloader
