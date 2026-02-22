"""
Simple Conditional Diffusion Model for Image Restoration
- Lightweight UNet with timestep embedding
- Conditional on degraded image
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embedding"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResBlock(nn.Module):
    """Residual block with timestep embedding"""
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x, t):
        h = self.norm1(F.silu(self.conv1(x)))
        h = h + self.time_mlp(t)[:, :, None, None]
        h = self.norm2(F.silu(self.conv2(h)))
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)
        
    def forward(self, x, t):
        h = self.res(x, t)
        return self.down(h), h


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res = ResBlock(in_ch + out_ch, out_ch, time_dim)  # concat skip
        
    def forward(self, x, skip, t):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t)


class SimpleUNet(nn.Module):
    """
    Lightweight UNet for Diffusion
    - Input: noisy image + condition (degraded image)
    - Output: predicted noise
    """
    def __init__(self, in_ch=6, out_ch=3, base_ch=64, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Initial conv
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        
        # Encoder
        self.down1 = DownBlock(base_ch, base_ch * 2, time_dim)      # 64 -> 128
        self.down2 = DownBlock(base_ch * 2, base_ch * 4, time_dim)  # 128 -> 256
        self.down3 = DownBlock(base_ch * 4, base_ch * 8, time_dim)  # 256 -> 512
        
        # Middle
        self.mid = ResBlock(base_ch * 8, base_ch * 8, time_dim)
        
        # Decoder
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, time_dim)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, time_dim)
        self.up1 = UpBlock(base_ch * 2, base_ch, time_dim)
        
        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, out_ch, 3, padding=1)
        )
        
    def forward(self, x, condition, t):
        """
        x: noisy image [B, 3, H, W]
        condition: degraded image [B, 3, H, W]
        t: timestep [B]
        """
        # Concat condition
        x = torch.cat([x, condition], dim=1)  # [B, 6, H, W]
        
        # Time embedding
        t_emb = timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Initial
        h = self.init_conv(x)
        
        # Encoder
        h, s1 = self.down1(h, t_emb)
        h, s2 = self.down2(h, t_emb)
        h, s3 = self.down3(h, t_emb)
        
        # Middle
        h = self.mid(h, t_emb)
        
        # Decoder
        h = self.up3(h, s3, t_emb)
        h = self.up2(h, s2, t_emb)
        h = self.up1(h, s1, t_emb)
        
        return self.out_conv(h)


class SimpleDiffusion(nn.Module):
    """
    Conditional Diffusion Model for Image Restoration
    - DDPM training
    - DDIM sampling
    """
    def __init__(self, 
                 base_ch=64,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        
        # UNet
        self.unet = SimpleUNet(in_ch=6, out_ch=3, base_ch=base_ch)
        
        # Beta schedule (linear)
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        
    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: add noise to clean image"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise, noise
    
    def training_loss(self, degraded, clean):
        """
        Compute training loss
        degraded: input degraded image [B, 3, H, W]
        clean: target clean image [B, 3, H, W]
        """
        B = degraded.shape[0]
        device = degraded.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=device)
        
        # Add noise to clean image
        noise = torch.randn_like(clean)
        noisy, _ = self.q_sample(clean, t, noise)
        
        # Predict noise
        noise_pred = self.unet(noisy, degraded, t)
        
        # MSE loss on noise
        loss = F.mse_loss(noise_pred, noise)
        return loss
    
    @torch.no_grad()
    def ddim_sample(self, degraded, num_steps=50):
        """DDIM sampling for fast inference"""
        device = degraded.device
        B = degraded.shape[0]
        
        # Start from pure noise
        x = torch.randn_like(degraded)
        
        # DDIM timesteps (evenly spaced)
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.unet(x, degraded, t_batch)
            
            # DDIM update
            alpha = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)
            
            # Predicted x0
            x0_pred = (x - torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(alpha)
            x0_pred = x0_pred.clamp(-1, 1)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_prev) * noise_pred
            
            # x_{t-1}
            x = torch.sqrt(alpha_prev) * x0_pred + dir_xt
        
        return x
    
    def forward(self, degraded, num_inference_steps=50):
        """
        Inference: restore degraded image
        """
        if self.training:
            raise RuntimeError("Use training_loss() for training")
        return self.ddim_sample(degraded, num_steps=num_inference_steps)
