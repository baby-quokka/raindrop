import gc
import torch
import torch.nn.functional as F

# diffusers 0.32+ uses torch.xpu, torch.mps. 구버전/비-Apple PyTorch 호환용 shim
class _DeviceShim:
    @staticmethod
    def empty_cache():
        pass
    @staticmethod
    def device_count():
        return 0
    @staticmethod
    def manual_seed(seed):
        torch.manual_seed(seed)
    @staticmethod
    def is_available():
        return False
    @staticmethod
    def reset_peak_memory_stats():
        pass
    @staticmethod
    def max_memory_allocated(*args, **kwargs):
        return 0
    @staticmethod
    def synchronize():
        pass

if not hasattr(torch, "xpu"):
    torch.xpu = _DeviceShim()
if not hasattr(torch, "mps"):
    torch.mps = _DeviceShim()

from tqdm import tqdm
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler
)

class RectifiedFlow(object):
    def __init__(self,
                 device="cpu",
                 model_id='sd2-community/stable-diffusion-2-1',
                 scheduler_id="stabilityai/stable-diffusion-3.5-medium"
                 ):
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(self.device)
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_id, subfolder='scheduler')
        self.unet, self.tokenizer, self.text_encoder = self.pipe.unet, self.pipe.tokenizer, self.pipe.text_encoder

        self.sigmas = self.scheduler.sigmas.to(self.device)
        self.timesteps = self.scheduler.timesteps.to(self.device)
        self.num_train_timesteps = self.scheduler.num_train_timesteps
        self.num_inference_steps = 50

        del self.pipe
        gc.collect()
        torch.cuda.empty_cache()

    def set_train(self):
        self.vae.requires_grad_(False), self.text_encoder.requires_grad_(False)
        self.vae.eval(), self.text_encoder.eval()
        self.unet.requires_grad_(True)
        self.unet.train()

    def set_eval(self):
        self.vae.requires_grad_(False), self.text_encoder.requires_grad_(False)
        self.vae.eval(), self.text_encoder.eval()
        self.unet.requires_grad_(False)
        self.unet.eval()

    def encode_img(self,
                   img
                   ):
        posterior = self.vae.encode(img).latent_dist
        return posterior.mode() * self.vae.config.scaling_factor
    
    def decode_latent(self,
                      latent
                      ):
        latent = latent / self.vae.config.scaling_factor
        img = self.vae.decode(latent).sample
        return img
    
    def encode_text(self,
                    text,
                    ):
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embedding = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        return text_embedding
    
    def get_sigmas(self,
                   timesteps,
                   ndim=4
                   ):
        timesteps = timesteps.to(self.device)
        indices = [(self.timesteps == t).nonzero().item() for t in timesteps]
        sigmas = self.sigmas[indices].flatten()
        while len(sigmas.shape) < ndim:
            sigmas = sigmas.unsqueeze(-1)
        return sigmas
    
    def flow_matching_loss(self,
                           x0,
                           x1,
                           text
                           ):
        with torch.no_grad():
            z0 = self.encode_img(x0)
            z1 = self.encode_img(x1)
            text_embedding = self.encode_text(text)

        time_indices = torch.randint(
            0, self.num_train_timesteps, (x0.size(0),)
        )
        timesteps = self.timesteps[time_indices].to(self.device)
        sigmas = self.get_sigmas(timesteps, ndim=x0.ndim)

        zt = sigmas * z1 + (1.0 - sigmas) * z0

        velocity_pred = self.unet(
            zt, timesteps, encoder_hidden_states=text_embedding
        ).sample
        velocity_target = (z1 - z0)

        loss = F.mse_loss(velocity_pred, velocity_target, reduction='mean')
        return loss
    
    @torch.no_grad()
    def inference(self,
                  x1,
                  text,
                  num_inference_steps=50
                  ):
        
        z1 = self.encode_img(x1)
        text_embedding = self.encode_text(text)

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        zt = z1
        pbar = tqdm(total=num_inference_steps, desc="Sampling")
        for timestep in timesteps:
            timestep = timestep.expand(x1.size(0))

            velocity_pred = self.unet(
                zt, timestep, encoder_hidden_states=text_embedding
            ).sample

            zt = self.scheduler.step(velocity_pred, timestep, zt, return_dict=False)[0]
            pbar.update(1)
        pbar.close()

        with torch.no_grad():
            x0 = self.decode_latent(zt)
        return x0

        

    

        
    


