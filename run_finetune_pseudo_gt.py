"""
B안: Pseudo-GT 생성 + 파인튜닝을 한 번에 실행하는 스크립트

사용법:
    python run_finetune_pseudo_gt.py \
        --model_name ConvIR_base \
        --checkpoint results/EXP_NAME/ckpt/Final.pkl \
        --valid_txt data/valid/RaindropClarity.txt \
        --pseudo_gt_dir pseudo_gt_output \
        --exp_suffix finetune_pseudo_gt \
        --num_iter 10000 \
        --learning_rate 1e-5
"""

import os
import sys
import argparse
import subprocess


def run_command(cmd, description):
    """명령어를 실행하고 결과를 출력합니다."""
    print(f"\n{'='*60}")
    print(f"[Step] {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"\n[Error] {description} failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\n[Success] {description} completed")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="B안: Pseudo-GT 생성 + 파인튜닝을 한 번에 실행"
    )
    
    # 공통 인자
    parser.add_argument("--model_name", type=str, required=True,
                        help="모델 이름 (예: ConvIR_base, Restormer)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="사전 학습된 모델 체크포인트 경로")
    parser.add_argument("--valid_txt", type=str, default="data/valid/RaindropClarity.txt",
                        help="입력 이미지 리스트 txt 경로")
    
    # Pseudo-GT 생성 관련
    parser.add_argument("--pseudo_gt_dir", type=str, default="pseudo_gt_output",
                        help="Pseudo-GT 저장 디렉토리")
    parser.add_argument("--pseudo_gt_save_mode", type=str, default="per_image",
                        choices=["per_scene", "per_image"],
                        help="Pseudo-GT 저장 방식")
    parser.add_argument("--scene_ranges_file", type=str, default="data/valid/RaindropClarity_scene_ranges.txt",
                        help="Scene 구간 파일 (A안과 동일). Pseudo-GT 생성 시 같은 풍경끼리 묶을 때 사용")
    parser.add_argument("--use_self_ensemble_for_pseudo_gt", action="store_true",
                        help="Pseudo-GT 생성 시 self-ensemble 사용")
    
    # 파인튜닝 관련
    parser.add_argument("--exp_suffix", type=str, default="finetune_pseudo_gt",
                        help="실험 이름 suffix")
    parser.add_argument("--num_iter", type=int, default=10000,
                        help="파인튜닝 iteration 수")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate (파인튜닝은 작은 값 권장, 기본 1e-5)")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help="Learning rate scheduler (기본: constant, 옵션: constant_with_warmup)")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="LR warmup steps (기본: 0)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--crop", type=int, default=256,
                        help="Crop size")
    parser.add_argument("--num_worker", type=int, default=4,
                        help="DataLoader worker 수")
    
    # Loss 가중치
    parser.add_argument("--lambda_l1", type=float, default=1.0,
                        help="L1 loss 가중치 (기본: 1.0)")
    parser.add_argument("--lambda_lpips", type=float, default=0.1,
                        help="LPIPS loss 가중치 (기본: 0.1, 기존 학습에서는 0.05/0.02/0.1 사용)")
    parser.add_argument("--lambda_msssim", type=float, default=0.1,
                        help="MS-SSIM loss 가중치 (기본: 0.1)")
    parser.add_argument("--lambda_fft", type=float, default=0.01,
                        help="FFT loss 가중치 (기본: 0.01)")
    parser.add_argument("--use_lpips", action="store_true",
                        help="LPIPS loss 사용")
    
    # 기타
    parser.add_argument("--no_augment", action="store_true",
                        help="STRRNet 권장: 파인튜닝 시 geometric aug 비활성화 (center crop만)")
    parser.add_argument("--skip_pseudo_gt", action="store_true",
                        help="Pseudo-GT 생성 단계 건너뛰기 (이미 생성된 경우)")
    parser.add_argument("--skip_finetune", action="store_true",
                        help="파인튜닝 단계 건너뛰기")
    parser.add_argument("--cuda_device", type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES 설정 (예: 6)")
    
    args = parser.parse_args()
    
    # CUDA_VISIBLE_DEVICES 설정
    env = os.environ.copy()
    if args.cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        print(f"[Info] Using GPU: {args.cuda_device}")
    
    # ==========================
    # Step 1: Pseudo-GT 생성
    # ==========================
    if not args.skip_pseudo_gt:
        pseudo_gt_cmd = [
            sys.executable, "generate_pseudo_gt.py",
            "--model_name", args.model_name,
            "--checkpoint", args.checkpoint,
            "--valid_txt", args.valid_txt,
            "--output_dir", args.pseudo_gt_dir,
            "--save_mode", args.pseudo_gt_save_mode,
        ]
        if getattr(args, "scene_ranges_file", None) and os.path.isfile(args.scene_ranges_file):
            pseudo_gt_cmd.extend(["--scene_ranges_file", args.scene_ranges_file])
        if args.use_self_ensemble_for_pseudo_gt:
            pseudo_gt_cmd.append("--use_self_ensemble")
        
        run_command(pseudo_gt_cmd, "Step 1: Pseudo-GT 생성")
    else:
        print("\n[Skip] Pseudo-GT 생성 단계 건너뛰기")
        if not os.path.exists(args.pseudo_gt_dir):
            print(f"[Error] Pseudo-GT 디렉토리가 존재하지 않습니다: {args.pseudo_gt_dir}")
            sys.exit(1)
    
    # ==========================
    # Step 2: 파인튜닝
    # ==========================
    if not args.skip_finetune:
        finetune_cmd = [
            sys.executable, "finetune_with_pseudo_gt.py",
            "--model_name", args.model_name,
            "--checkpoint", args.checkpoint,
            "--input_list", args.valid_txt,
            "--pseudo_gt_dir", args.pseudo_gt_dir,
            "--exp_suffix", args.exp_suffix,
            "--num_iter", str(args.num_iter),
            "--learning_rate", str(args.learning_rate),
            "--lr_scheduler", args.lr_scheduler,
            "--lr_warmup_steps", str(args.lr_warmup_steps),
            "--batch_size", str(args.batch_size),
            "--crop", str(args.crop),
            "--num_worker", str(args.num_worker),
            "--lambda_l1", str(args.lambda_l1),
            "--lambda_lpips", str(args.lambda_lpips),
            "--lambda_msssim", str(args.lambda_msssim),
            "--lambda_fft", str(args.lambda_fft),
        ]
        
        if args.use_lpips:
            finetune_cmd.append("--use_lpips")
        if getattr(args, "no_augment", False):
            finetune_cmd.append("--no_augment")
        
        run_command(finetune_cmd, "Step 2: 파인튜닝")
    else:
        print("\n[Skip] 파인튜닝 단계 건너뛰기")
    
    print("\n" + "="*60)
    print("[Complete] B안 실행 완료!")
    print("="*60)
    print("\n다음 단계: 파인튜닝된 모델로 추론 실행")
    print(f"  python inference_ntire_dev.py \\")
    print(f"    --model_name {args.model_name} \\")
    print(f"    --checkpoint experiment/{args.model_name}-{args.exp_suffix}-XXXX/ckpt/Final.pkl \\")
    print(f"    --output_dir submission/YOUR_OUTPUT_DIR \\")
    print(f"    --use_multi_view_fusion \\")
    print(f"    --use_self_ensemble")
    print()


if __name__ == "__main__":
    main()
