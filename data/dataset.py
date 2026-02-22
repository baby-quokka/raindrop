import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from data.utils import *

class IRDataset(Dataset):
    def __init__(self, 
                 filename: str="data/train/RaindropClarity.txt",
                 crop: int = 256,
                 is_test: bool = False,
                 paired: bool = True,
                 value_range = (0, 1)
                 ):
        super().__init__()
        self.filename = filename
        self.crop = crop
        self.is_test = is_test
        self.paired = paired
        self.value_range = value_range
        self.data = os.path.splitext(os.path.basename(self.filename))[0]

        # 주어진 파일(self.filename)에서 각 줄(이미지 경로)을 읽어 리스트로 저장
        # 빈 줄은 제외하고, 앞뒤 공백도 제거
        with open(self.filename, "r") as f:
            self.input_files: List[str] = [ln.strip() for ln in f if ln.strip()]

        self.files = []
        for input_path in self.input_files:
            if "RaindropClarity" in self.data or "/Drop/" in input_path:
                gt_path = input_path.replace('/Drop/', '/Clear/')
            elif "NH-HAZE" in self.data or "/hazy" in input_path:
                dirpath, fname = input_path.rsplit('/', 1)
                gt_dir = dirpath.replace("/input", "/target")
                img_id = fname.replace('hazy', 'GT')
                gt_path = f"{gt_dir}/{img_id}"
            else:
                # 기본: 동일 경로 사용 (unpaired 또는 알 수 없는 데이터셋)
                gt_path = input_path
            self.files.append((input_path, gt_path))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        input_path, gt_path = self.files[idx]
        a = read_rgb(input_path)  # uint8 [3,H,W]
        if self.paired:
            b = read_rgb(gt_path)  # uint8 [3,H,W]
        
        if not self.is_test:
            a, b = crop_pair(a, b, self.crop)
            a, b = flip_pair(a, b, p=0.5)
            if self.value_range == (0, 1):
                return to_m01(a), to_m01(b)
            elif self.value_range == (-1, 1):
                return to_m11(a), to_m11(b)
            else:
                raise ValueError(f"Unsupported value_range {self.value_range}")
        else:
            name = os.path.splitext(os.path.basename(input_path))[0]
            if self.value_range == (0, 1):
                if self.paired:
                    return to_m01(a), to_m01(b), name
                else:
                    return to_m01(a), name
            elif self.value_range == (-1, 1):
                if self.paired:
                    return to_m11(a), to_m11(b), name
                else:
                    return to_m11(a), name
            else:
                raise ValueError(f"Unsupported value_range {self.value_range}")

def train_dataloader(filename="data/train/RaindropClarity.txt", value_range=(0, 1), crop=256, batch_size=64, num_workers=4):
    dataloader = DataLoader(
        IRDataset(filename, crop=crop, value_range=value_range),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    return dataloader

def test_dataloader(filename="data/test/RaindropClarity.txt", value_range=(0, 1), paired=False, batch_size=1, num_workers=1):
    dataloader = DataLoader(
        IRDataset(filename, is_test=True, value_range=value_range, paired=paired),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )
    return dataloader

def valid_dataloader(filename="data/valid/RaindropClarity.txt", value_range=(0, 1), paired=False, batch_size=1, num_workers=1):
    dataloader = DataLoader(
        IRDataset(filename, is_test=True, value_range=value_range, paired=paired),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )
    return dataloader