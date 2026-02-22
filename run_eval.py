"""
임시 Evaluation 스크립트
학습 완료된 모델 평가
"""
import torch
import argparse
import models
import os
from eval import _eval

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--test_model', type=str, required=True)
    parser.add_argument('--test_data', type=str, default='RaindropClarity')
    parser.add_argument('--test_txt', type=str, default='', help='Custom test txt path (overrides test_data)')
    parser.add_argument('--value_range', type=tuple, default=(0, 1))
    parser.add_argument('--save_image', type=bool, default=True)
    parser.add_argument('--use_self_ensemble', action='store_true',
                        help='Use x8 self-ensemble at test time')
    parser.add_argument('--max_save_images', type=int, default=-1,
                        help='최대 저장할 이미지 개수 (-1 이면 전체 저장)')
    args = parser.parse_args()
    
    print(f"=" * 60)
    print(f"Evaluating {args.model_name}")
    print(f"Model: {args.test_model}")
    print(f"Test Data: {args.test_data}")
    print(f"=" * 60)
    
    # 모델 생성
    model_class = getattr(models, args.model_name)
    model = model_class()
    
    # 체크포인트 로드 (.pkl=state_dict만 / .pt=dict일 수 있음)
    # BasicSR 형식: params, params_ema / 일반: model / raw: state_dict
    ckpt = torch.load(args.test_model, map_location='cpu')
    if isinstance(ckpt, dict):
        checkpoint = ckpt.get('model') or ckpt.get('params_ema') or ckpt.get('params') or ckpt
    else:
        checkpoint = ckpt
    model.load_state_dict(checkpoint, strict=False)
    model = model.cuda()
    
    # submission/EXP_NAME 폴더에 결과 저장 + readme.txt 생성
    args.result_dir = os.path.join('submission', args.exp_name)
    args.summary_txt_name = 'readme.txt'
    
    # 평가 실행
    _eval(model, args)
    
    print(f"Evaluation complete for {args.model_name}")


if __name__ == '__main__':
    main()
