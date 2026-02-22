import os
import random
from typing import Tuple

import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

def ceil_m(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def read_rgb(path: str) -> torch.Tensor:
    """
    Fast path: uint8 CHW, always 3-channel RGB (alpha dropped if exists).
    """
    x = read_image(path, mode=ImageReadMode.RGB)  # uint8 [3,H,W]
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"Bad image shape: {path} -> {tuple(x.shape)}")
    return x


def to_m11(x_u8: torch.Tensor) -> torch.Tensor:
    """
    uint8 [3,H,W] -> float32 [-1,1]
    """
    x = x_u8.float().div_(255.0)
    return x.mul_(2.0).sub_(1.0)

def to_m01(x_u8: torch.Tensor) -> torch.Tensor:
    """
    uint8 [3,H,W] -> float32 [0,1]
    """
    x = x_u8.float().div_(255.0)
    return x


def maybe_resize(x_u8: torch.Tensor, crop: int, m: int, interp: InterpolationMode) -> torch.Tensor:
    """
    (2) rule:
      - if short >= crop: return as-is
      - else:
          1) isotropic resize so short >= crop (ceil to avoid short=crop-1 due to rounding)
          2) anisotropic resize ONLY along long axis so long becomes multiple of m
    """
    _, h, w = x_u8.shape
    short = h if h < w else w
    if short >= crop:
        return x_u8

    scale = crop / short
    new_h = int(h * scale + 0.999999)  # ceil-ish
    new_w = int(w * scale + 0.999999)

    x_u8 = TF.resize(x_u8, [new_h, new_w], interpolation=interp, antialias=True)

    _, h2, w2 = x_u8.shape
    if h2 >= w2:
        th, tw = ceil_m(h2, m), w2
    else:
        th, tw = h2, ceil_m(w2, m)

    if th == h2 and tw == w2:
        return x_u8

    return TF.resize(x_u8, [th, tw], interpolation=interp, antialias=True)

def crop_pair(a: torch.Tensor, b: torch.Tensor, crop: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Paired random crop.
    - crop = int  -> (crop, crop)
    - crop = (ch, cw) or [ch, cw]
    Assumes H,W >= crop size.
    """
    if isinstance(crop, int):
        ch, cw = crop, crop
    else:
        if len(crop) != 2:
            raise ValueError(f"crop must be int or (h, w), got: {crop}")
        ch, cw = int(crop[0]), int(crop[1])

    _, h, w = a.shape
    if h < ch or w < cw:
        raise ValueError(f"After preprocess, image is smaller than crop: {(h, w)} < {(ch, cw)}")

    if h == ch and w == cw:
        return a, b

    i = torch.randint(0, h - ch + 1, (1,), device=a.device).item()
    j = torch.randint(0, w - cw + 1, (1,), device=a.device).item()

    return a[:, i:i + ch, j:j + cw], b[:, i:i + ch, j:j + cw]


def center_crop_pair(a: torch.Tensor, b: torch.Tensor, crop: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Paired center crop (deterministic, no random). STRRNet-style finetune 시 geometric aug 비활성화용."""
    if isinstance(crop, int):
        ch, cw = crop, crop
    else:
        ch, cw = int(crop[0]), int(crop[1])
    _, h, w = a.shape
    if h < ch or w < cw:
        raise ValueError(f"Image smaller than crop: {(h, w)} < {(ch, cw)}")
    i = (h - ch) // 2
    j = (w - cw) // 2
    return a[:, i:i + ch, j:j + cw], b[:, i:i + ch, j:j + cw]


def flip_pair(a: torch.Tensor, b: torch.Tensor, p: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    if random.random() < p:
        # flip W dimension
        a = a.flip(-1)
        b = b.flip(-1)
    return a, b


# ============================
# Stage1 전용 유틸 (Drop/Blur/Clear 3장 동시 변환)
# ============================
def crop_triple(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, crop: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paired random crop for three images (e.g. Drop, Blur, Clear)."""
    if isinstance(crop, int):
        ch, cw = crop, crop
    else:
        ch, cw = int(crop[0]), int(crop[1])
    _, h, w = a.shape
    if h < ch or w < cw:
        raise ValueError(f"Image smaller than crop: {(h, w)} < {(ch, cw)}")
    if h == ch and w == cw:
        return a, b, c
    i = torch.randint(0, h - ch + 1, (1,), device=a.device).item()
    j = torch.randint(0, w - cw + 1, (1,), device=a.device).item()
    return (
        a[:, i : i + ch, j : j + cw],
        b[:, i : i + ch, j : j + cw],
        c[:, i : i + ch, j : j + cw],
    )


def flip_triple(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, p: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paired horizontal flip for three images."""
    if random.random() < p:
        a, b, c = a.flip(-1), b.flip(-1), c.flip(-1)
    return a, b, c