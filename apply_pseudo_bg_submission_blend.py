import os
import argparse
from typing import List, Tuple, Dict, Optional

import shutil

import torch
from torchvision.utils import save_image
from torchvision import transforms as T
from PIL import Image


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


def build_start_to_end(ranges: List[Tuple[int, int]]) -> Dict[int, int]:
    """scene_ranges.txt 에서 start -> end 매핑 생성."""
    return {s: e for s, e in ranges}


def build_val_to_test_mapping() -> Dict[Tuple[int, int], Tuple[int, int, str]]:
    """
    validation bg scene 구간 -> test scene 구간 및 val pseudo 이름 매핑.

    반환:
      (val_start, val_end) -> (test_start, test_end, val_scene_tag)
    """
    mapping: Dict[Tuple[int, int], Tuple[int, int, str]] = {}

    # 137-152  (validation 00320~00340 : bg) -> test drop
    mapping[(320, 340)] = (137, 152, "val_bg_00320_00340")

    # 157-172  (validation 00341~00373 : bg) -> test drop
    mapping[(341, 373)] = (157, 172, "val_bg_00341_00373")

    # 382-397  (validation 00001~00020 : bg) -> test drop
    mapping[(1, 20)] = (382, 397, "val_bg_00001_00020")

    # 620-631  (validation 00148~00184 : bg) -> test drop
    mapping[(148, 184)] = (620, 631, "val_bg_00148_00184")

    # 632-651  (validation 00185~00210 : bg) -> test drop
    mapping[(185, 210)] = (632, 651, "val_bg_00185_00210")

    # 692-707  (validation 00211~00239 : bg) -> test drop
    mapping[(211, 239)] = (692, 707, "val_bg_00211_00239")

    return mapping


def build_test_pseudo_mapping(start_to_end: Dict[int, int],
                              test_pseudo_dir: str) -> Dict[Tuple[int, int], str]:
    """
    test pseudo_bg 파일 경로 매핑 생성.

    - independent_starts: 완전 독립 scene
    - bgpair_starts: bg/drop 짝이 있는 bg focus scene
    """
    # 255-270 (drop) <-> 355-370 (bg) 는
    # bg 구간(355-370)에서 만든 pseudo 를 두 구간에 모두 적용할 것이므로
    # independent_starts 에서 255, 355 를 빼고 별도 매핑을 만든다.
    independent_starts = [1, 21, 41, 61, 79, 97, 117, 177, 275,
                          422, 462, 482, 500, 520, 540]
    bgpair_starts = [295, 315, 335, 442, 560, 580, 600]

    mapping: Dict[Tuple[int, int], str] = {}

    # 일반 independent scene 들
    for s in independent_starts:
        if s not in start_to_end:
            continue
        e = start_to_end[s]
        scene_name = f"test_indep_{s:05d}_{e:05d}.png"
        mapping[(s, e)] = os.path.join(test_pseudo_dir, scene_name)

    # bg/drop pair scene 들
    for s in bgpair_starts:
        if s not in start_to_end:
            continue
        e = start_to_end[s]
        scene_name = f"test_bgpair_{s:05d}_{e:05d}.png"
        mapping[(s, e)] = os.path.join(test_pseudo_dir, scene_name)

    # 255-270 (drop) <-> 355-370 (bg) 특별 처리:
    # 355-370 구간에서 만든 pseudo (파일명은 test_indep_00355_00370.png) 를
    # 두 구간 모두에 사용한다.
    if 255 in start_to_end and 355 in start_to_end:
        e_drop = start_to_end[255]
        e_bg = start_to_end[355]
        bg_scene_name = f"test_indep_{355:05d}_{e_bg:05d}.png"
        bg_pseudo_path = os.path.join(test_pseudo_dir, bg_scene_name)
        mapping[(255, e_drop)] = bg_pseudo_path
        mapping[(355, e_bg)] = bg_pseudo_path

    return mapping


def find_range(num: int, ranges: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    for s, e in ranges:
        if s <= num <= e:
            return s, e
    return None


def load_image_as_tensor(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return T.ToTensor()(img).unsqueeze(0)  # [1, 3, H, W], range [0,1]


def blend_and_save(baseline_path: str, pseudo_path: str, out_path: str, alpha: float):
    """
    alpha: baseline 비율 (0.2이면 20% baseline + 80% pseudo)
    """
    base = load_image_as_tensor(baseline_path)
    pseudo = load_image_as_tensor(pseudo_path)

    # 크기가 다를 경우 pseudo 를 baseline 크기로 resize
    if base.shape[2:] != pseudo.shape[2:]:
        pseudo_img = T.ToPILImage()(pseudo.squeeze(0))
        pseudo_img = pseudo_img.resize((base.shape[3], base.shape[2]), Image.BILINEAR)
        pseudo = T.ToTensor()(pseudo_img).unsqueeze(0)

    blended = alpha * base + (1.0 - alpha) * pseudo
    blended = torch.clamp(blended, 0.0, 1.0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(blended, out_path, normalize=False)


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) test 입력 리스트 로드
    test_paths = load_paths(args.test_txt)

    # 2) scene range 정보 로드
    scene_ranges = load_scene_ranges(args.scene_ranges_file)
    start_to_end = build_start_to_end(scene_ranges)

    # 3) validation pseudo_bg -> test 범위 매핑
    val_to_test = build_val_to_test_mapping()
    # test 범위 -> val pseudo 파일 경로 매핑으로 변환
    test_from_val: Dict[Tuple[int, int], str] = {}
    for (_, _), (ts, te, val_tag) in val_to_test.items():
        test_from_val[(ts, te)] = os.path.join(args.val_pseudo_dir, f"{val_tag}.png")

    # 4) test 자체에서 만든 pseudo_bg 매핑
    test_pseudo = build_test_pseudo_mapping(start_to_end, args.test_pseudo_dir)

    # 5) 각 test 이미지에 대해 최종 출력 생성
    for in_path in test_paths:
        num = num_from_path(in_path)
        if num < 0:
            continue

        base_name = f"{num:05d}.png"
        baseline_path = os.path.join(args.baseline_dir, base_name)
        if not os.path.isfile(baseline_path):
            print(f"[Warning] baseline not found for {base_name}: {baseline_path}")
            continue

        # (1) validation 기반 pseudo 가 있는 scene인가?
        pseudo_path = None
        r = find_range(num, list(test_from_val.keys()))
        if r is not None:
            pseudo_path = test_from_val[r]

        # (2) 아니면 test pseudo_bg 가 있는 scene인가?
        if pseudo_path is None:
            r2 = find_range(num, list(test_pseudo.keys()))
            if r2 is not None:
                pseudo_path = test_pseudo[r2]

        out_path = os.path.join(args.output_dir, base_name)

        # (3) pseudo 가 있으면 alpha blending, 없으면 baseline 그대로 복사
        if pseudo_path is not None and os.path.isfile(pseudo_path):
            try:
                blend_and_save(baseline_path, pseudo_path, out_path, alpha=args.alpha)
            except Exception as e:
                print(f"[Error] blending failed for {base_name}: {e}, fallback to baseline copy")
                shutil.copyfile(baseline_path, out_path)
        else:
            shutil.copyfile(baseline_path, out_path)

    print(f"[Info] Final blended submission saved to: {os.path.abspath(args.output_dir)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Apply pseudo BG with alpha blending to build final submission")
    parser.add_argument("--test_txt", type=str, default="data/final/RaindropClarity_test-input.txt")
    parser.add_argument("--scene_ranges_file", type=str, default="data/final/RaindropClarity_scene_ranges.txt")
    parser.add_argument("--val_pseudo_dir", type=str, default="pseudo_bg/val")
    parser.add_argument("--test_pseudo_dir", type=str, default="pseudo_bg/test")
    parser.add_argument("--baseline_dir", type=str, required=True,
                        help="기본 ConvIR baseline 결과 폴더 (예: ConvIR_large-final-baseline)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="최종 제출 이미지가 저장될 폴더")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="baseline 비율 (0.2 = 20% baseline + 80% pseudo)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

