import torch
import datetime
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import argparse
import models
from train import _train
from eval import _eval
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, ProjectConfiguration

def main(args):

    if args.mode == 'train':
        now = datetime.datetime.now()
        if getattr(args, 'exp_suffix', None):
            args.exp_name = f"{args.model_name}-{args.exp_suffix}-{args.train_data}-{now.strftime('%m%d-%H%M')}"
        else:
            args.exp_name = f"{args.model_name}-{args.train_data}-{now.strftime('%m%d-%H%M')}"
        project_config = ProjectConfiguration(
            project_dir=f"experiment/{args.exp_name}",
            logging_dir=f"experiment/{args.exp_name}/logs"
        )
        # STRRNet 등 일부 모델은 conditional forward로 미사용 파라미터가 생길 수 있음
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            log_with="wandb" if args.use_wandb else None,
            project_config=project_config,
            gradient_accumulation_steps=args.grad_accum,
            kwargs_handlers=[ddp_kwargs]
        )

        set_seed(args.seed)
        model_class = getattr(models, args.model_name)
        model = model_class()

        _train(model, accelerator, args)

    elif args.mode == 'test':
        model_class = getattr(models, args.model_name)
        model = model_class()
        checkpoint = torch.load(args.test_model, map_location='cpu')
        model.load_state_dict(checkpoint)
        model = model.cuda()
        _eval(model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='ConvIR_large', type=str)
    parser.add_argument('--train_data', type=str, default='RaindropClarity')
    parser.add_argument('--test_data', type=str, default='RaindropClarity')
    parser.add_argument('--exp_suffix', type=str, default=None, help='Experiment suffix for results folder, e.g. 05, 02, ema, 05ema → model-05-RaindropClarity-...')
    parser.add_argument('--value_range', type=tuple, default=(0, 1))
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--seed', type=int, default=42)

    # Train
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    parser.add_argument('--crop', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument('--num_iter', type=int, default=100000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--valid_freq', type=int, default=5000)

    parser.add_argument('--use_lpips', action='store_true', help='Use LPIPS loss')
    parser.add_argument('--lambda_l1', type=float, default=1.0, help='Weight for L1 loss')
    parser.add_argument('--lambda_lpips', type=float, default=0.1)
    parser.add_argument('--lambda_msssim', type=float, default=0.1, help='Weight for MS-SSIM loss (1 - MS-SSIM)')
    parser.add_argument('--lambda_fft', type=float, default=0.01, help='Weight for FFT magnitude L1 loss (raw scale ~6-7, so 0.01 recommended)')
    parser.add_argument('--use_ema', action='store_true', help='Use EMA (decay=0.999) for valid/save/inference')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')
    parser.add_argument('--use_wandb', action='store_true', help='Use WandB for logging (requires wandb login)')

    # Resume
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume training')

    # Test
    parser.add_argument('--test_model', type=str, default='')
    parser.add_argument('--test_txt', type=str, default='', help='Custom test txt path (overrides test_data)')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])
    parser.add_argument('--use_self_ensemble', action='store_true', help='Use x8 self-ensemble at test time')

    args = parser.parse_args()
    print(args)

    # 명령줄 인자로 받은 mode에 따라 한 가지만 실행
    if args.mode == 'train':
        main(args)
    elif args.mode == 'test':
        main(args)
