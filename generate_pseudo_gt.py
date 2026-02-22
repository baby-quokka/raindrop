"""
B안: Pseudo-GT 생성 스크립트

같은 풍경의 여러 raindrop 이미지에서 median fusion을 통해 
pseudo ground truth를 생성합니다.

이 pseudo-GT는 이후 파인튜닝에 사용됩니다.

사용법:
    python generate_pseudo_gt.py \
        --checkpoint path/to/model.pkl \
        --valid_txt data/valid/RaindropClarity.txt \
        --output_dir pseudo_gt_output \
        --model_name Restormer
"""

import os
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

import models
from data import valid_dataloader
from utils import norm_range
from eval import forward_x8_self_ensemble
from inference_ntire_dev import (
    build_model,
    median_fusion,
    _load_scene_ranges,
    _build_scene_groups,
)


@torch.no_grad()
def generate_pseudo_gt(args):
    """
    같은 scene의 여러 이미지에서 median fusion을 통해 pseudo-GT를 생성합니다.
    
    생성된 pseudo-GT는:
    - 각 scene마다 하나의 fused 이미지 생성
    - 또는 각 입력 이미지에 대해 해당 scene의 fused 결과를 저장
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Info] Using device: {device}")

    # 모델 로드
    print(f"[Info] Loading model {args.model_name} from {args.checkpoint}")
    model = build_model(args.model_name, args.checkpoint, device)

    # 출력 폴더 준비
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Pseudo-GT 저장 방식 선택
    # - per_scene: 각 scene마다 하나의 fused 이미지 (scene_id.png)
    # - per_image: 각 입력 이미지마다 해당 scene의 fused 결과 저장
    save_mode = getattr(args, "save_mode", "per_image")

    is_uformer = model.__class__.__name__ == "Uformer"
    factor = 32

    with open(args.valid_txt, "r") as f:
        image_paths = [ln.strip() for ln in f if ln.strip()]

    # Scene별로 그룹화 (A안과 동일한 구간 사용)
    scene_ranges = None
    scene_ranges_file = getattr(args, "scene_ranges_file", None)
    if scene_ranges_file and os.path.isfile(scene_ranges_file):
        scene_ranges = _load_scene_ranges(scene_ranges_file)
        print(f"[Info] Loaded {len(scene_ranges)} scene ranges from {scene_ranges_file}")
    
    scene_group_size = getattr(args, "scene_group_size", 0)
    scene_groups = _build_scene_groups(image_paths, scene_group_size, scene_ranges or [])
    
    print(f"[Info] Found {len(scene_groups)} scenes, {len(image_paths)} total images")
    
    # 각 scene별 이미지 개수 통계
    scene_sizes = [len(paths) for paths in scene_groups.values()]
    print(f"[Info] Images per scene: min={min(scene_sizes)}, max={max(scene_sizes)}, avg={sum(scene_sizes)/len(scene_sizes):.1f}")

    # 데이터 로더 생성
    dataloader = valid_dataloader(
        filename=args.valid_txt,
        value_range=(0, 1),
        paired=False,
        batch_size=1,
        num_workers=1,
    )
    
    # 경로를 키로 하는 딕셔너리 생성
    path_to_data = {}
    for data in dataloader:
        input_img, name = data
        if isinstance(name, (list, tuple)):
            base_name = name[0]
        else:
            base_name = name
        
        # 원본 경로 찾기
        for path in image_paths:
            if base_name in path or os.path.basename(path) == f"{base_name}.png":
                path_to_data[path] = (input_img, base_name)
                break

    # Scene별로 처리
    for scene_id, paths in tqdm(scene_groups.items(), desc="Generating pseudo-GT"):
        scene_outputs = []
        scene_names = []
        
        # 같은 scene의 모든 이미지에 대해 추론 수행
        for path in paths:
            if path not in path_to_data:
                continue
                
            input_img, base_name = path_to_data[path]
            input_img = input_img.to(device)
            
            h, w = input_img.shape[2], input_img.shape[3]
            
            # 패딩
            if is_uformer:
                max_size = max(h, w)
                H = W = ((max_size + factor) // factor) * factor
            else:
                H = ((h + factor) // factor) * factor
                W = ((w + factor) // factor) * factor
            
            padh = H - h
            padw = W - w
            input_pad = F.pad(input_img, (0, padw, 0, padh), "reflect")
            
            # 추론
            if getattr(args, "use_self_ensemble", False):
                pred = forward_x8_self_ensemble(model, input_pad)
            else:
                pred = model(input_pad)
                if isinstance(pred, (list, tuple)):
                    pred = pred[-1]
            pred = pred[:, :, :h, :w]
            pred = norm_range(pred, value_range=(0, 1))
            
            scene_outputs.append(pred)
            scene_names.append(base_name)
        
        if len(scene_outputs) == 0:
            continue
        
        # Median fusion으로 pseudo-GT 생성
        if len(scene_outputs) > 1:
            pseudo_gt = median_fusion(scene_outputs)
        else:
            pseudo_gt = scene_outputs[0]
        
        # 저장
        if save_mode == "per_scene":
            # 각 scene마다 하나의 fused 이미지 저장
            save_path = os.path.join(args.output_dir, f"scene_{scene_id}.png")
            save_image(pseudo_gt, save_path, normalize=True, value_range=(0, 1))
        else:
            # per_image: 각 입력 이미지에 대해 해당 scene의 fused 결과 저장
            # (파인튜닝 시 원본 입력과 매칭하기 쉽도록)
            for base_name in scene_names:
                save_path = os.path.join(args.output_dir, f"{base_name}.png")
                save_image(pseudo_gt, save_path, normalize=True, value_range=(0, 1))
    
    print(f"[Info] Pseudo-GT saved to: {os.path.abspath(args.output_dir)}")
    
    # 파인튜닝용 파일 리스트 생성 (선택사항)
    if getattr(args, "generate_finetune_list", False):
        finetune_list_path = os.path.join(args.output_dir, "finetune_list.txt")
        with open(finetune_list_path, "w") as f:
            for path in image_paths:
                base_name = os.path.splitext(os.path.basename(path))[0]
                pseudo_gt_path = os.path.join(args.output_dir, f"{base_name}.png")
                if os.path.exists(pseudo_gt_path):
                    f.write(f"{path} {pseudo_gt_path}\n")
        print(f"[Info] Finetune list saved to: {finetune_list_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pseudo-GT using median fusion")
    
    parser.add_argument("--model_name", type=str, default="Restormer",
                        help="모델 이름")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="학습된 모델 체크포인트 경로")
    parser.add_argument("--valid_txt", type=str, default="data/valid/RaindropClarity.txt",
                        help="입력 이미지 리스트 txt 경로")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Pseudo-GT 저장 폴더")
    parser.add_argument("--save_mode", type=str, default="per_image",
                        choices=["per_scene", "per_image"],
                        help="저장 방식: per_scene(각 scene마다 하나) 또는 per_image(각 이미지마다)")
    parser.add_argument("--scene_ranges_file", type=str, default="",
                        help="Scene 구간 파일 (A안과 동일). 예: data/valid/RaindropClarity_scene_ranges.txt")
    parser.add_argument("--scene_group_size", type=int, default=0,
                        help="scene_ranges_file 미사용 시, 연속 N장을 한 scene으로 묶을 때 사용")
    parser.add_argument("--use_self_ensemble", action="store_true",
                        help="x8 self-ensemble 사용")
    parser.add_argument("--generate_finetune_list", action="store_true",
                        help="파인튜닝용 파일 리스트 생성")
    parser.add_argument("--cpu", action="store_true",
                        help="CPU 사용")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_pseudo_gt(args)
