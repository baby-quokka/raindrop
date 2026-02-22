"""
NTIRE RaindropClarity dev 추론 스크립트.

[A안] Multi-view fusion (학습 없이 추론 단에서만 적용)
  - 같은 풍경(scene)으로 묶인 이미지들에 대해:
    1) 각 이미지를 모델로 복원 → pred_1, pred_2, ...
    2) scene별 median fusion → median = median(pred_1, pred_2, ...)
    3) 최종 출력 (STRRNet 방식): 최종_i = alpha * pred_i + (1 - alpha) * median
    4) 입력 이미지 1장당 최종 1장씩 저장 (총 406장)
  - Scene 구간: --scene_ranges_file 로 지정 (예: data/valid/RaindropClarity_scene_ranges.txt)
  - alpha 기본값 0.7 (--fusion_alpha 로 변경 가능, 출처: STRRNet 구현 기본값)

  예시 명령어 (A안):
    CUDA_VISIBLE_DEVICES=6 python inference_ntire_dev.py \\
      --model_name ConvIR_base \\
      --checkpoint results/.../Final.pkl \\
      --output_dir submission/ConvIR-base-l1ema200k-fusionA \\
      --use_multi_view_fusion \\
      --use_self_ensemble \\
      --fusion_alpha 0.7 \\
      --scene_ranges_file data/valid/RaindropClarity_scene_ranges.txt
"""
import os
import time
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


def build_model(model_name: str, checkpoint: str, device: torch.device) -> torch.nn.Module:
    """모델 생성 후 체크포인트 로드."""
    model_class = getattr(models, model_name)
    model = model_class()

    ckpt = torch.load(checkpoint, map_location="cpu")
    # B안 파인튜닝 등은 dict로 저장됨: {'model': state_dict, 'optimizer': ..., ...}
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and ("params_ema" in ckpt or "params" in ckpt):
        state_dict = ckpt.get("params_ema") or ckpt.get("params")
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys: {len(missing)} (e.g. {list(missing)[:3]}...)")
    if unexpected:
        print(f"[Warning] Unexpected keys: {len(unexpected)} (e.g. {list(unexpected)[:3]}...)")

    model = model.to(device)
    model.eval()
    return model


def extract_scene_id(path: str) -> str:
    """
    경로에서 scene ID를 추출합니다.
    
    RaindropClarity 데이터셋 구조:
    - /Drop/00124/00005.png -> scene_id: 00124
    - /Drop/00001.png -> scene_id: 00001 (파일명 기반)
    """
    parts = path.replace('\\', '/').split('/')
    
    # /Drop/00124/00005.png 형식인 경우
    for i, part in enumerate(parts):
        if part == 'Drop' and i + 1 < len(parts):
            # 다음 부분이 숫자로만 이루어진 폴더명이면 scene_id
            next_part = parts[i + 1]
            if next_part.isdigit() and len(next_part) == 5:  # 5자리 숫자 폴더
                return next_part
    
    # 파일명 기반으로 추출 (예: 00001.png -> 00001)
    filename = os.path.basename(path)
    scene_id = os.path.splitext(filename)[0]
    
    # 파일명이 숫자로만 이루어진 경우, 앞 5자리를 scene_id로 사용
    # (같은 풍경의 여러 이미지가 연속된 번호를 가질 수 있음)
    if scene_id.isdigit():
        # 예: 00001, 00002, 00003 -> 모두 같은 scene으로 간주하거나
        # 또는 일정 범위로 그룹화 (예: 00001-00010 -> scene 00001)
        # 여기서는 간단하게 파일명의 앞부분을 scene_id로 사용
        # 사용자가 원하면 더 정교한 그룹화 로직 추가 가능
        return scene_id
    
    return os.path.dirname(path)


def median_fusion(images: list) -> torch.Tensor:
    """
    여러 이미지에 대해 median fusion을 적용합니다.
    
    같은 풍경의 여러 raindrop 이미지에서 배경은 일관되지만 
    물방울 위치가 다르므로, median fusion으로 일관된 배경을 추출합니다.
    
    Args:
        images: [1, 3, H, W] 형태의 텐서 리스트
    
    Returns:
        [1, 3, H, W] median-fused 이미지
    """
    if len(images) == 1:
        return images[0]
    
    # 모든 이미지의 크기가 같다고 가정
    stacked = torch.cat(images, dim=0)  # [N, 3, H, W]
    median, _ = torch.median(stacked, dim=0, keepdim=True)
    return median


def weighted_fusion(original: torch.Tensor, median: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    """
    원본 출력과 median-fused 이미지를 가중치 합으로 융합합니다.
    
    Args:
        original: [1, 3, H, W] 모델의 개별 출력
        median: [1, 3, H, W] median-fused 이미지
        alpha: original에 대한 가중치 (0.7 = 70% original + 30% median)
    
    Returns:
        [1, 3, H, W] 최종 융합된 이미지
    """
    return alpha * original + (1 - alpha) * median


def adaptive_fusion(outputs: list, input_images: list, alpha: float = 0.7) -> list:
    """
    여러 출력을 적응적으로 융합합니다.
    
    각 픽셀 위치에서:
    - 여러 이미지 중 가장 일관된(신뢰할 수 있는) 값을 선택
    - 또는 median과 개별 출력을 가중치 합
    
    Args:
        outputs: [1, 3, H, W] 형태의 모델 출력 리스트
        input_images: [1, 3, H, W] 형태의 입력 이미지 리스트
        alpha: 개별 출력에 대한 가중치
    
    Returns:
        각 입력에 대한 융합된 출력 리스트
    """
    if len(outputs) == 1:
        return outputs
    
    # Median fusion으로 일관된 배경 추출
    median_output = median_fusion(outputs)
    
    # 각 출력에 대해 weighted fusion 적용
    fused_outputs = []
    for output in outputs:
        fused = weighted_fusion(output, median_output, alpha=alpha)
        fused_outputs.append(fused)
    
    return fused_outputs


@torch.no_grad()
def run_inference(args):
    """
    NTIRE RaindropClarity dev(= Drop 폴더 407장)용 추론 스크립트.
    - 입력 리스트: data/valid/RaindropClarity.txt
    - 출력 폴더: submission (디폴트)
    - 출력 파일명: 입력 파일명과 동일 (예: 00001.png)
    - readme.txt: NTIRE 포맷으로 자동 생성
    
    Multi-view fusion 옵션:
    - 같은 풍경의 여러 raindrop 이미지에서 배경 정보를 융합하여 복원 품질 개선
    - 큰 물방울이나 물줄기로 가려진 부분이 다른 이미지에서는 잘 보일 수 있음
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Info] Using device: {device}")

    # 모델 로드
    print(f"[Info] Loading model {args.model_name} from {args.checkpoint}")
    model = build_model(args.model_name, args.checkpoint, device)

    # 출력 폴더 준비 (submission)
    os.makedirs(args.output_dir, exist_ok=True)

    is_uformer = model.__class__.__name__ == "Uformer"
    factor = 32

    # Multi-view fusion 사용 여부 확인
    use_multi_view_fusion = getattr(args, "use_multi_view_fusion", False)
    
    if use_multi_view_fusion:
        # Scene별로 그룹화하여 처리
        print("[Info] Using multi-view fusion: grouping images by scene")
        _run_inference_with_fusion(model, args, device, is_uformer, factor)
    else:
        # 기존 방식: 개별 이미지 처리
        print("[Info] Processing images individually")
        _run_inference_individual(model, args, device, is_uformer, factor)


@torch.no_grad()
def _run_inference_individual(model, args, device, is_uformer, factor):
    """기존 방식: 각 이미지를 개별적으로 처리"""
    dataloader = valid_dataloader(
        filename=args.valid_txt,
        value_range=(0, 1),
        paired=False,
        batch_size=1,
        num_workers=1,
    )

    times = []

    for data in tqdm(dataloader, desc="Inference (NTIRE dev)"):
        # paired=False 이므로 (input, name) 형태
        input_img, name = data
        input_img = input_img.to(device)

        # name 이 list/tuple 로 들어올 수 있음
        if isinstance(name, (list, tuple)):
            base_name = name[0]
        else:
            base_name = name

        h, w = input_img.shape[2], input_img.shape[3]

        # Uformer는 정사각형 패딩, 나머지는 32 배수 패딩 (eval.py 와 동일 로직)
        if is_uformer:
            max_size = max(h, w)
            H = W = ((max_size + factor) // factor) * factor
        else:
            H = ((h + factor) // factor) * factor
            W = ((w + factor) // factor) * factor

        padh = H - h
        padw = W - w
        input_pad = F.pad(input_img, (0, padw, 0, padh), "reflect")

        start_t = time.time()
        if getattr(args, "use_self_ensemble", False):
            pred = forward_x8_self_ensemble(model, input_pad)
        else:
            pred = model(input_pad)
            if isinstance(pred, (list, tuple)):
                pred = pred[-1]
        pred = pred[:, :, :h, :w]
        pred = norm_range(pred, value_range=(0, 1))
        elapsed = time.time() - start_t
        times.append(elapsed)

        # max_save_images: -1이면 전체, 0 이상이면 해당 개수만 저장
        max_save = getattr(args, "max_save_images", -1)
        if max_save < 0 or len(times) <= max_save:
            save_path = os.path.join(args.output_dir, f"{base_name}.png")
            save_image(pred, save_path, normalize=True, value_range=(0, 1))

    # 평균 시간 계산 및 readme.txt 생성
    _save_readme(args, times)


def _load_scene_ranges(filepath: str) -> list:
    """파일에서 scene 구간 읽기. 한 줄에 'start end' (포함)."""
    ranges = []
    with open(filepath, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) >= 2:
                start, end = int(parts[0]), int(parts[1])
                ranges.append((start, end))
    return ranges


def _image_number_from_path(path: str) -> int:
    """경로/파일명에서 이미지 번호 추출 (예: 00001.png -> 1). 없으면 -1."""
    base = os.path.splitext(os.path.basename(path))[0]
    return int(base) if base.isdigit() else -1


def _build_scene_groups(image_paths: list, scene_group_size: int, scene_ranges: list):
    """
    - scene_ranges가 있으면: (start, end) 구간으로 scene 할당. (322번 공백 등 누락 번호는 해당 구간에 없으면 단일 scene 처리)
    - scene_group_size > 0 이면: 연속 N장씩 한 scene.
    - 그 외: 경로에서 extract_scene_id로 그룹화.
    """
    if scene_ranges:
        scene_groups = defaultdict(list)
        for path in image_paths:
            num = _image_number_from_path(path)
            if num < 0:
                scene_groups["_other"].append(path)
                continue
            scene_id = None
            for idx, (start, end) in enumerate(scene_ranges):
                if start <= num <= end:
                    scene_id = str(idx)
                    break
            if scene_id is None:
                scene_id = f"_orphan_{num}"
            scene_groups[scene_id].append(path)
        # 같은 scene 내에서는 이미지 번호 순 정렬
        for sid in scene_groups:
            scene_groups[sid].sort(key=_image_number_from_path)
        return scene_groups

    if scene_group_size and scene_group_size > 0:
        def sort_key(p):
            return (_image_number_from_path(p) if _image_number_from_path(p) >= 0 else 0, p)
        sorted_paths = sorted(image_paths, key=sort_key)
        scene_groups = defaultdict(list)
        for idx, path in enumerate(sorted_paths):
            scene_id = str(idx // scene_group_size)
            scene_groups[scene_id].append(path)
        return scene_groups

    scene_groups = defaultdict(list)
    for path in image_paths:
        scene_id = extract_scene_id(path)
        scene_groups[scene_id].append(path)
    return scene_groups


@torch.no_grad()
def _run_inference_with_fusion(model, args, device, is_uformer, factor):
    """
    Multi-view fusion을 사용한 추론.
    같은 scene의 이미지들을 그룹화하여 배경 정보를 융합합니다.
    """
    with open(args.valid_txt, "r") as f:
        image_paths = [ln.strip() for ln in f if ln.strip()]
    
    scene_ranges = None
    scene_ranges_file = getattr(args, "scene_ranges_file", None)
    if scene_ranges_file and os.path.isfile(scene_ranges_file):
        scene_ranges = _load_scene_ranges(scene_ranges_file)
        print(f"[Info] Loaded {len(scene_ranges)} scene ranges from {scene_ranges_file}")
    
    scene_group_size = getattr(args, "scene_group_size", 0)
    scene_groups = _build_scene_groups(image_paths, scene_group_size, scene_ranges or [])
    
    print(f"[Info] Found {len(scene_groups)} scenes, {len(image_paths)} total images")
    
    scene_sizes = [len(paths) for paths in scene_groups.values()]
    print(f"[Info] Images per scene: min={min(scene_sizes)}, max={max(scene_sizes)}, avg={sum(scene_sizes)/len(scene_sizes):.1f}")
    
    if max(scene_sizes) == 1:
        print("[Warning] 모든 scene에 이미지가 1장뿐이라 fusion 효과가 없습니다. "
              "--scene_group_size 5 또는 10 등으로 연속 N장을 한 scene으로 묶어보세요.")
    
    # 데이터 로더 생성 (전체 이미지)
    dataloader = valid_dataloader(
        filename=args.valid_txt,
        value_range=(0, 1),
        paired=False,
        batch_size=1,
        num_workers=1,
    )
    
    # 경로를 키로 하는 딕셔너리 생성 (빠른 조회용)
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
    
    times = []
    processed_count = 0
    
    # Scene별로 처리
    for scene_id, paths in tqdm(scene_groups.items(), desc="Processing scenes"):
        scene_outputs = []
        scene_inputs = []
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
            
            start_t = time.time()
            if getattr(args, "use_self_ensemble", False):
                pred = forward_x8_self_ensemble(model, input_pad)
            else:
                pred = model(input_pad)
                if isinstance(pred, (list, tuple)):
                    pred = pred[-1]
            pred = pred[:, :, :h, :w]
            pred = norm_range(pred, value_range=(0, 1))
            elapsed = time.time() - start_t
            times.append(elapsed)
            
            scene_outputs.append(pred)
            scene_inputs.append(input_img)
            scene_names.append(base_name)
        
        # Multi-view fusion 적용
        fusion_alpha = getattr(args, "fusion_alpha", 0.7)
        if len(scene_outputs) > 1:
            fused_outputs = adaptive_fusion(scene_outputs, scene_inputs, alpha=fusion_alpha)
        else:
            fused_outputs = scene_outputs
        
        # 결과 저장
        max_save = getattr(args, "max_save_images", -1)
        for base_name, pred in zip(scene_names, fused_outputs):
            if max_save < 0 or processed_count < max_save:
                save_path = os.path.join(args.output_dir, f"{base_name}.png")
                save_image(pred, save_path, normalize=True, value_range=(0, 1))
            processed_count += 1
    
    # 평균 시간 계산 및 readme.txt 생성
    _save_readme(args, times)


def _save_readme(args, times):
    """readme.txt 파일을 생성합니다."""
    avg_time = sum(times) / len(times) if times else 0.0
    print(f"[Info] Measured average runtime per img: {avg_time:.4f} s")

    runtime_to_write = avg_time if args.runtime_per_img <= 0 else args.runtime_per_img

    readme_path = os.path.join(args.output_dir, "readme.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"runtime per img [s] : {runtime_to_write:.4f}\n")
        f.write("CPU[1] / GPU[0] : 0\n")
        f.write("Extra Data [1] / No Extra Data [0] : 0\n")
        f.write(f"Other description : {args.other_description}\n")

    print(f"[Info] Submission saved to: {os.path.abspath(args.output_dir)}")
    print(f"[Info] readme.txt created at: {readme_path}")



def parse_args():
    parser = argparse.ArgumentParser(description="NTIRE RaindropClarity dev inference")

    # 모델 및 체크포인트
    parser.add_argument("--model_name", type=str, default="Restormer",
                        help="models/__init__.py 에 등록된 모델 이름 (예: Restormer, Uformer 등)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="학습된 모델 가중치(.pkl) 경로 (예: results/EXP_NAME/ckpt/Final.pkl)")

    # 입력 리스트(txt) & 출력 폴더
    parser.add_argument(
        "--valid_txt",
        type=str,
        default="data/valid/RaindropClarity.txt",
        help="NTIRE dev 이미지 리스트 txt 경로",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="submission",
        help="NTIRE 제출용 결과 이미지 및 readme.txt 가 저장될 폴더",
    )

    # 환경 설정
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="강제로 CPU 사용 (기본은 GPU 사용 가능 시 GPU)",
    )
    parser.add_argument(
        "--use_self_ensemble",
        action="store_true",
        help="x8 self-ensemble (rotation 0/90/180/270 x hflip on/off) at test time",
    )
    parser.add_argument(
        "--use_multi_view_fusion",
        action="store_true",
        help="같은 풍경의 여러 raindrop 이미지에서 배경 정보를 융합하여 복원 품질 개선",
    )
    parser.add_argument(
        "--fusion_alpha",
        type=float,
        default=0.7,
        help="Multi-view fusion에서 개별 출력에 대한 가중치 (0.7 = 70%% 개별 + 30%% median)",
    )
    parser.add_argument(
        "--scene_group_size",
        type=int,
        default=0,
        help="0이면 경로에서 scene 추출. >0이면 리스트 순서대로 연속 N장을 한 scene으로 묶음",
    )
    parser.add_argument(
        "--scene_ranges_file",
        type=str,
        default="",
        help="Scene 구간 파일 경로. 한 줄에 'start end' (포함). 지정 시 이걸로 scene 그룹 생성 (예: data/valid/RaindropClarity_scene_ranges.txt)",
    )
    parser.add_argument(
        "--max_save_images",
        type=int,
        default=-1,
        help="최대 저장할 이미지 개수 (-1이면 전체 저장)",
    )

    # readme.txt 내용 (사용자 요구 포맷)
    # 기본은 "측정된 평균 시간"을 쓰고 싶으니,
    # 0 이하로 두면 측정값 사용, 양수로 주면 강제 오버라이드.
    parser.add_argument(
        "--runtime_per_img",
        type=float,
        default=0.0,
        help="> 0 이면 readme.txt에 이 값을 강제로 사용, <= 0 이면 측정된 평균 시간 사용",
    )
    parser.add_argument(
        "--other_description",
        type=str,
        default="",
        help="readme.txt 의 Other description 내용",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)

