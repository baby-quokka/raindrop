import os
import lpips
import wandb
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import argparse
import datetime
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
from models import RectifiedFlow
from torchvision.utils import save_image, make_grid
from data import train_dataloader, test_dataloader
from diffusers.optimization import get_scheduler
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, ProjectConfiguration

import transformers
import diffusers
transformers.logging.set_verbosity_error()
diffusers.logging.set_verbosity_error()

from huggingface_hub import login

# Hugging Face 토큰은 코드에 하드코딩하지 말고 환경변수로 받기
# 예) export HF_TOKEN=... (bash) / setx HF_TOKEN ... (PowerShell)
hf_token = os.getenv("HF_TOKEN", "").strip()
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not set. If this is a gated model, login may be required.")

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=str, default="NTIRE2026")
parser.add_argument('--data', type=str, default='NH-HAZE')
parser.add_argument('--model_id', type=str, default='sd2-community/stable-diffusion-2-1')
parser.add_argument('--scheduler_id', type=str, default='stabilityai/stable-diffusion-3.5-medium')

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--crop', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--num_iter', type=int, default=30000)
parser.add_argument('--log_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=5000)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--lr_scheduler', type=str, default="constant")
parser.add_argument('--lr_warmup_steps', type=int, default=1000)
parser.add_argument('--lr_num_cycles', type=int, default=1)
parser.add_argument('--lr_power', type=float, default=1)

parser.add_argument('--mixed_precision', type=str, default='no')
parser.add_argument('--grad_accum', type=int, default=4)
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--wandb_num_images', type=int, default=8, help='Number of visualization images to log to wandb per save step')
args = parser.parse_args()

set_seed(args.seed)

exp_name = f"RectifiedFlow-{args.data}-{datetime.datetime.now().strftime('%m%d-%H%M')}"
project_config = ProjectConfiguration(
    project_dir=f"experiment/{exp_name}",
    logging_dir=f"experiment/{exp_name}/logs"
)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
accelerator = Accelerator(
    mixed_precision=args.mixed_precision,
    gradient_accumulation_steps=args.grad_accum,
    project_config=project_config,
    kwargs_handlers=[ddp_kwargs],
    log_with="wandb"
)
if accelerator.is_main_process:
    wandb.init(project=args.project, name=exp_name, config=vars(args))
    ckpt_save_dir = os.path.join("experiment", exp_name, "checkpoints")
    vis_save_dir = os.path.join("experiment", exp_name, "visualizations")
    os.makedirs(ckpt_save_dir, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)

train_loader = train_dataloader(
    filename=f"data/train/{args.data}.txt",
    crop=args.crop, value_range=(-1, 1),
    batch_size=args.batch_size, num_workers=args.num_workers
)

if args.data == "RaindropClarity":
    paired = False
if args.data == "NH-HAZE":
    paired = True

test_loader = test_dataloader(
    filename=f"data/test/{args.data}.txt",
    value_range=(-1, 1), paired=paired,
    batch_size=1, num_workers=1
)

rectified_flow = RectifiedFlow(
    device=accelerator.device, 
    model_id=args.model_id,
    scheduler_id=args.scheduler_id
)
rectified_flow.set_train()

optimizer = torch.optim.AdamW(rectified_flow.unet.parameters(), lr=args.lr)
scheduler = get_scheduler(
    args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps,
    num_training_steps=args.num_iter, num_cycles=args.lr_num_cycles, power=args.lr_power
)

rectified_flow.unet, train_loader, optimizer, scheduler = accelerator.prepare(
    rectified_flow.unet, train_loader, optimizer, scheduler
)

lpips_fn = lpips.LPIPS(net='vgg').to(accelerator.device)

pbar = tqdm(range(args.num_iter), disable=not accelerator.is_main_process, 
    desc=f"Training Diffusion Model with {args.data} dataset")
iter_loader = iter(train_loader)
global_iter, iter_loss_adder, iter_adder = 0, 0, 0

while global_iter < args.num_iter:
    batch = next(iter_loader, None)
    if batch is None:
        iter_loader = iter(train_loader)
        batch = next(iter_loader, None)

    input_img, label_img = batch
    input_img = input_img.to(accelerator.device, non_blocking=True)
    label_img = label_img.to(accelerator.device, non_blocking=True)

    with accelerator.accumulate(rectified_flow.unet):
        if args.data == "RaindropClarity":
            loss = rectified_flow.flow_matching_loss(x0=label_img, x1=input_img, text=[""]*input_img.size(0))
        if args.data == "NH-HAZE":
            loss = rectified_flow.flow_matching_loss(x0=input_img, x1=label_img, text=[""]*input_img.size(0))
        accelerator.backward(loss)
        iter_loss_adder += loss.item()
        iter_adder += 1

        if accelerator.sync_gradients:
            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(rectified_flow.unet.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_iter += 1
            if accelerator.is_main_process:
                pbar.update(1)
            if (global_iter % args.log_freq) == 0:
                local_avg_loss = iter_loss_adder / iter_adder
                avg_loss_tensor = torch.tensor(local_avg_loss).to(accelerator.device)
                global_avg_loss = accelerator.reduce(avg_loss_tensor, reduction="mean").item()
                if accelerator.is_main_process:
                    wandb.log({"Train/Loss": global_avg_loss, 
                               "Learning Rate": optimizer.param_groups[0]["lr"]}, step=global_iter)
                iter_loss_adder, iter_adder = 0, 0
            if (global_iter % args.save_freq) == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ckpt_save_name = os.path.join(ckpt_save_dir, f"model_{global_iter}.pkl")
                    torch.save(accelerator.unwrap_model(rectified_flow.unet).state_dict(), ckpt_save_name)

                    vis_save_dir = os.path.join("experiment", exp_name, "visualizations", f"iter_{global_iter}")
                    os.makedirs(vis_save_dir, exist_ok=True)
                    with torch.no_grad():
                        # WandB에 올릴 샘플을 너무 많이 만들지 않도록 제한
                        wandb_imgs = []
                        wandb_cap = int(getattr(args, "wandb_num_images", 8))
                        logged = 0
                        for batch in test_loader:
                            if args.data == "RaindropClarity":
                                input_img, img_name = batch
                                input_img = input_img.to(accelerator.device)

                                with torch.no_grad():
                                    pred_img = rectified_flow.inference(x1=input_img, text=[""]*input_img.size(0))
                            elif args.data == "NH-HAZE":
                                input_img, label_img, img_name = batch
                                input_img = input_img.to(accelerator.device)
                                label_img = label_img.to(accelerator.device)
                                label_img = TF.center_crop(label_img, [256, 256])

                                with torch.no_grad():
                                    pred_img = rectified_flow.inference(x1=label_img, text=[""]*input_img.size(0))
                                pred_img = torch.cat([label_img, pred_img], dim=3)

                            vis_save_name = os.path.join(vis_save_dir, f"{img_name[0]}.png")
                            save_image(pred_img, vis_save_name, normalize=True, value_range=(-1, 1))

                            # WandB: 일부만 이미지로 기록 (pred_img는 [-1,1], normalize해서 [0,1]로)
                            if logged < wandb_cap:
                                img01 = (pred_img.detach().cpu() + 1.0) / 2.0
                                # batch=1 가정. 혹시 batch가 커도 첫 장만
                                grid = make_grid(img01[:1], nrow=1)  # [3,H,W] 또는 [3,H,2W]
                                wandb_imgs.append(wandb.Image(grid, caption=f"{img_name[0]}"))
                                logged += 1

                        if wandb_imgs:
                            wandb.log({"Valid/Samples": wandb_imgs}, step=global_iter)
                accelerator.wait_for_everyone()
if accelerator.is_main_process:
    ckpt_save_name = os.path.join(ckpt_save_dir, f"model_final.pkl")
    torch.save(accelerator.unwrap_model(rectified_flow.unet).state_dict(), ckpt_save_name)
pbar.close()
wandb.finish()   
accelerator.end_training()         





