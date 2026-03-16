import os
import argparse
from typing import List, Tuple, Dict

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


def load_paths(txt_path: str) -> List[str]:
    with open(txt_path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def num_from_path(path: str) -> int:
    base = os.path.splitext(os.path.basename(path))[0]
    return int(base) if base.isdigit() else -1


def load_scene_ranges(path: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            a, b = ln.split()[:2]
            ranges.append((int(a), int(b)))
    return ranges


def build_scene_map(ranges: List[Tuple[int, int]]) -> Dict[int, Tuple[int, int, int]]:
    """
    start 번호 -> (start, end, scene_idx) 매핑 생성.
    scene_idx 는 0-based 인덱스.
    """
    m: Dict[int, Tuple[int, int, int]] = {}
    for idx, (s, e) in enumerate(ranges):
        m[s] = (s, e, idx)
    return m


@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Info] Using device: {device}")

    model = build_model(args.model_name, args.checkpoint, device)

    all_paths = load_paths(args.test_txt)
    scene_ranges = load_scene_ranges(args.scene_ranges_file)
    start_to_range = build_scene_map(scene_ranges)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 완전 독립 scene (final test 전용)
    independent_starts = [1, 21, 41, 61, 79, 97, 117, 177, 255, 275, 355,
                          422, 462, 482, 500, 520, 540]

    # 2) test 내부 bg/drop 짝이 있는 bg focus scene
    bgpair_starts = [295, 315, 335, 442, 560, 580, 600]

    target_starts = sorted(set(independent_starts + bgpair_starts))

    factor = 32
    is_uformer = model.__class__.__name__ == "Uformer"

    for start in target_starts:
        if start not in start_to_range:
            print(f"[Warning] start {start} not found in scene_ranges_file, skip")
            continue
        s, e, scene_idx = start_to_range[start]

        # 이 scene 에 속하는 test-input 경로 모으기
        scene_paths = [p for p in all_paths
                       if (num_from_path(p) >= s and num_from_path(p) <= e)]
        scene_paths.sort(key=num_from_path)

        if not scene_paths:
            print(f"[Warning] No test paths for scene {s}-{e}")
            continue

        scene_tag = "indep" if start in independent_starts else "bgpair"
        scene_name = f"test_{scene_tag}_{s:05d}_{e:05d}"
        print(f"[Info] Processing scene {scene_name} ({s}-{e}), {len(scene_paths)} images")

        outputs: List[torch.Tensor] = []

        for p in tqdm(scene_paths, desc=f"Scene {scene_name}"):
            img = torch.from_numpy(
                __import__("imageio").v2.imread(p)
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

        fused = median_fusion(outputs)
        save_path = os.path.join(args.output_dir, f"{scene_name}.png")
        save_image(fused, save_path, normalize=True, value_range=(0, 1))
        print(f"[Info] Saved pseudo BG for {scene_name} to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Make pseudo BG for final test scenes")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_txt", type=str, default="data/final/RaindropClarity_test-input.txt")
    parser.add_argument("--scene_ranges_file", type=str, default="data/final/RaindropClarity_scene_ranges.txt")
    parser.add_argument("--output_dir", type=str, default="pseudo_bg/test")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use_self_ensemble", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

