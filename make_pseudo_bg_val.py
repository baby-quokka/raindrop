import os
import argparse
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

import models
from utils import norm_range
from eval import forward_x8_self_ensemble


def build_model(model_name: str, checkpoint: str, device: torch.device) -> torch.nn.Module:
    model_class = getattr(models, model_name)
    model = model_class()

    ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and ("params_ema" in ckpt or "params" in ckpt):
        state_dict = ckpt.get("params_ema") or ckpt.get("params")
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def median_fusion(images: List[torch.Tensor]) -> torch.Tensor:
    if len(images) == 1:
        return images[0]
    stacked = torch.cat(images, dim=0)
    median, _ = torch.median(stacked, dim=0, keepdim=True)
    return median


def load_val_paths(txt_path: str) -> List[str]:
    with open(txt_path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def num_from_path(path: str) -> int:
    base = os.path.splitext(os.path.basename(path))[0]
    return int(base) if base.isdigit() else -1


def select_range(paths: List[str], start_idx: int, end_idx: int) -> List[str]:
    selected = []
    for p in paths:
        n = num_from_path(p)
        if n < 0:
            continue
        if start_idx <= n <= end_idx:
            selected.append(p)
    selected.sort(key=num_from_path)
    return selected


@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Info] Using device: {device}")

    model = build_model(args.model_name, args.checkpoint, device)

    all_paths = load_val_paths(args.valid_txt)

    os.makedirs(args.output_dir, exist_ok=True)

    # validation bg focus 구간들 (포함 범위)
    # 00001~00020, 00148~00184, 00185~00210, 00211~00239, 00320~00340, 00341~00373
    scene_ranges: List[Tuple[int, int, str]] = [
        (1, 20, "val_bg_00001_00020"),
        (148, 184, "val_bg_00148_00184"),
        (185, 210, "val_bg_00185_00210"),
        (211, 239, "val_bg_00211_00239"),
        (320, 340, "val_bg_00320_00340"),
        (341, 373, "val_bg_00341_00373"),
    ]

    factor = 32
    is_uformer = model.__class__.__name__ == "Uformer"

    for start, end, scene_name in scene_ranges:
        paths = select_range(all_paths, start, end)
        if not paths:
            print(f"[Warning] No paths found for range {start}-{end}")
            continue

        print(f"[Info] Processing scene {scene_name} ({start}-{end}), {len(paths)} images")

        outputs = []
        for p in tqdm(paths, desc=f"Scene {scene_name}"):
            # validation txt에는 /root/dataset/ImageRestoration/... 절대 경로가 들어 있으므로
            # 그대로 사용한다.
            local_path = p

            img = torch.from_numpy(
                __import__("imageio").v2.imread(local_path)
            ).float().permute(2, 0, 1).unsqueeze(0) / 255.0

            img = img.to(device)
            h, w = img.shape[2], img.shape[3]

            if is_uformer:
                max_size = max(h, w)
                H = W = ((max_size + factor) // factor) * factor
            else:
                H = ((h + factor) // factor) * factor
                W = ((w + factor) // factor) * factor

            padh = H - h
            padw = W - w
            img_pad = F.pad(img, (0, padw, 0, padh), "reflect")

            if args.use_self_ensemble:
                pred = forward_x8_self_ensemble(model, img_pad)
            else:
                pred = model(img_pad)
                if isinstance(pred, (list, tuple)):
                    pred = pred[-1]
            pred = pred[:, :, :h, :w]
            pred = norm_range(pred, value_range=(0, 1))
            outputs.append(pred)

        if not outputs:
            continue

        fused = median_fusion(outputs)
        save_path = os.path.join(args.output_dir, f"{scene_name}.png")
        save_image(fused, save_path, normalize=True, value_range=(0, 1))
        print(f"[Info] Saved pseudo BG for {scene_name} to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Make pseudo BG for validation RaindropClarity scenes")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--valid_txt", type=str, default="data/valid/RaindropClarity.txt")
    parser.add_argument("--output_dir", type=str, default="pseudo_bg/val")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use_self_ensemble", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

