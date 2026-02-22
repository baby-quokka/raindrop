"""
DiffIR: Efficient Diffusion Model for Image Restoration
Paper: https://arxiv.org/abs/2303.09472 (CVPR 2023)

핵심 아이디어:
1. 2-Stage 구조: IR Prior → Diffusion Refinement
2. CIRP (Compact IR Prior): 가벼운 네트워크로 대략적 복원
3. DIRT (Dynamic IR Transformer): 동적 attention으로 정밀 복원
4. Residual Prediction: Diffusion이 clean image가 아닌 residual 예측

기존 Diffusion IR 대비 장점:
- 기존: Clean image 직접 생성 (1000 steps 필요)
- DiffIR: IR Prior 결과의 residual만 예측 (훨씬 적은 steps)

전체 흐름:
    Degraded Image
         ↓
    ┌─────────────┐
    │  IR Prior   │  → Rough restoration (x_prior)
    └─────────────┘
         ↓
    ┌─────────────┐
    │  Diffusion  │  → Residual prediction (r = x_clean - x_prior)
    │    DIRT     │
    └─────────────┘
         ↓
    x_output = x_prior + r
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


##########################################################################
## Helper Functions
##########################################################################

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Sinusoidal timestep embedding
    
    Transformer의 positional encoding과 유사
    Diffusion의 timestep을 연속적인 벡터로 변환
    
    Args:
        timesteps: [B] timestep indices
        dim: embedding dimension
        max_period: maximum period for sinusoids
    
    Returns:
        [B, dim] timestep embeddings
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


##########################################################################
## Layer Normalization
##########################################################################

class LayerNorm(nn.Module):
    """Channel-wise Layer Normalization for 4D tensors"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


##########################################################################
## CIRP (Compact IR Prior) Components
##########################################################################

class SimpleGate(nn.Module):
    """Gating mechanism: x1 * x2"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    NAFNet-style Block for IR Prior
    
    NAFNet (ECCV 2022)의 기본 블록 사용
    - Simple Gate 기반 비선형성
    - Depth-wise Conv로 local features
    - Channel Attention으로 global features
    
    구조:
        Input
          ↓
        LayerNorm → 1x1 Conv → Gate → DWConv → 1x1 Conv → + (residual)
          ↓
        LayerNorm → 1x1 Conv → Gate → 1x1 Conv → + (residual)
          ↓
        Output
    """
    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden = dim * expansion
        
        # Spatial Mixing
        self.norm1 = LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, hidden * 2, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.conv2 = nn.Conv2d(hidden, dim, 1)
        self.gate1 = SimpleGate()
        
        # Channel Mixing
        self.norm2 = LayerNorm(dim)
        self.conv3 = nn.Conv2d(dim, hidden * 2, 1)
        self.conv4 = nn.Conv2d(hidden, dim, 1)
        self.gate2 = SimpleGate()
        
        # Learnable scaling
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        
    def forward(self, x):
        # Spatial mixing
        residual = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.gate1(x)
        x = self.dwconv(x)
        x = self.conv2(x)
        x = residual + x * self.beta
        
        # Channel mixing
        residual = x
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.gate2(x)
        x = self.conv4(x)
        x = residual + x * self.gamma
        
        return x


class CIRP(nn.Module):
    """
    Compact IR Prior (CIRP)
    
    가벼운 네트워크로 대략적인 복원 수행
    NAFNet 구조 기반으로 효율적인 IR
    
    역할:
    - Diffusion의 부담 감소 (rough restoration 제공)
    - Clean image의 대략적인 구조 복원
    - Residual은 diffusion이 담당
    
    Args:
        in_ch: 입력 채널
        dim: 기본 채널 수
        num_blocks: NAFBlock 수
    """
    def __init__(self, in_ch=3, dim=64, num_blocks=8):
        super().__init__()
        
        self.intro = nn.Conv2d(in_ch, dim, 3, 1, 1)
        
        self.body = nn.Sequential(
            *[NAFBlock(dim) for _ in range(num_blocks)]
        )
        
        self.outro = nn.Conv2d(dim, in_ch, 3, 1, 1)
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] degraded image
        Returns:
            [B, 3, H, W] roughly restored image
        """
        feat = self.intro(x)
        feat = self.body(feat)
        out = self.outro(feat)
        return out + x  # Global residual


##########################################################################
## DIRT (Dynamic IR Transformer) Components
##########################################################################

class MDTA(nn.Module):
    """
    Multi-DConv Head Transposed Attention
    
    Restormer의 MDTA를 timestep conditioning과 함께 사용
    Transposed attention: channel-wise attention (O(C²) complexity)
    
    Args:
        dim: 채널 수
        num_heads: attention head 수
        time_dim: timestep embedding 차원
    """
    def __init__(self, dim, num_heads=8, time_dim=256):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        # Q, K, V projection with depth-wise conv
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3)
        
        # Output projection
        self.proj = nn.Conv2d(dim, dim, 1)
        
        # Timestep modulation
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2)
        )
        
    def forward(self, x, t_emb):
        """
        Args:
            x: [B, C, H, W] features
            t_emb: [B, time_dim] timestep embedding
        """
        B, C, H, W = x.shape
        
        # Timestep modulation (scale and shift)
        t = self.time_mlp(t_emb)
        scale, shift = t.chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        x = x * (1 + scale) + shift
        
        # Q, K, V with depth-wise conv
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        # L2 normalize Q, K
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # Transposed attention: [B, heads, C/heads, C/heads]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        # Apply attention
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        
        return self.proj(out)


class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network
    
    Restormer의 GDFN에 timestep conditioning 추가
    
    Args:
        dim: 채널 수
        expansion: 확장 비율
        time_dim: timestep embedding 차원
    """
    def __init__(self, dim, expansion=2.66, time_dim=256):
        super().__init__()
        hidden = int(dim * expansion)
        
        self.conv1 = nn.Conv2d(dim, hidden * 2, 1)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2)
        self.conv2 = nn.Conv2d(hidden, dim, 1)
        self.gate = SimpleGate()
        
        # Timestep modulation
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2)
        )
        
    def forward(self, x, t_emb):
        """
        Args:
            x: [B, C, H, W] features
            t_emb: [B, time_dim] timestep embedding
        """
        # Timestep modulation
        t = self.time_mlp(t_emb)
        scale, shift = t.chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        x = x * (1 + scale) + shift
        
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.gate(x)
        x = self.conv2(x)
        
        return x


class DIRTBlock(nn.Module):
    """
    Dynamic IR Transformer Block
    
    Restormer block + Timestep conditioning
    
    구조:
        Input + t_emb
          ↓
        LayerNorm → MDTA (with t_emb) → + (residual)
          ↓
        LayerNorm → GDFN (with t_emb) → + (residual)
          ↓
        Output
    """
    def __init__(self, dim, num_heads=8, expansion=2.66, time_dim=256):
        super().__init__()
        
        self.norm1 = LayerNorm(dim)
        self.attn = MDTA(dim, num_heads, time_dim)
        
        self.norm2 = LayerNorm(dim)
        self.ffn = GDFN(dim, expansion, time_dim)
        
    def forward(self, x, t_emb):
        x = x + self.attn(self.norm1(x), t_emb)
        x = x + self.ffn(self.norm2(x), t_emb)
        return x


##########################################################################
## DIRT U-Net
##########################################################################

class Downsample(nn.Module):
    """Downsample: 해상도 1/2, 채널 2배"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, 4, 2, 1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample: 해상도 2배, 채널 1/2"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim // 2, 4, 2, 1)
    
    def forward(self, x):
        return self.conv(x)


class DIRT(nn.Module):
    """
    Dynamic IR Transformer (DIRT)
    
    Diffusion을 위한 noise prediction network
    U-Net 구조 + Transformer blocks + Timestep conditioning
    
    입력:
    - noisy: 노이즈가 추가된 residual
    - condition: IR Prior 결과
    - timestep: diffusion timestep
    
    출력:
    - 예측된 noise
    
    Args:
        in_ch: 입력 채널 (noisy + condition = 6)
        out_ch: 출력 채널 (3, noise)
        dim: 기본 채널 수
        num_blocks: 각 level의 transformer block 수
        num_heads: attention head 수
        time_dim: timestep embedding 차원
    """
    def __init__(self, in_ch=6, out_ch=3, dim=64, 
                 num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8],
                 time_dim=256):
        super().__init__()
        
        self.time_dim = time_dim
        
        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Input conv
        self.conv_in = nn.Conv2d(in_ch, dim, 3, 1, 1)
        
        # Encoder
        self.enc1 = nn.ModuleList([
            DIRTBlock(dim, num_heads[0], time_dim=time_dim) 
            for _ in range(num_blocks[0])])
        self.down1 = Downsample(dim)
        
        self.enc2 = nn.ModuleList([
            DIRTBlock(dim * 2, num_heads[1], time_dim=time_dim) 
            for _ in range(num_blocks[1])])
        self.down2 = Downsample(dim * 2)
        
        self.enc3 = nn.ModuleList([
            DIRTBlock(dim * 4, num_heads[2], time_dim=time_dim) 
            for _ in range(num_blocks[2])])
        self.down3 = Downsample(dim * 4)
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            DIRTBlock(dim * 8, num_heads[3], time_dim=time_dim) 
            for _ in range(num_blocks[3])])
        
        # Decoder
        self.up3 = Upsample(dim * 8)
        self.reduce3 = nn.Conv2d(dim * 8, dim * 4, 1)
        self.dec3 = nn.ModuleList([
            DIRTBlock(dim * 4, num_heads[2], time_dim=time_dim) 
            for _ in range(num_blocks[2])])
        
        self.up2 = Upsample(dim * 4)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, 1)
        self.dec2 = nn.ModuleList([
            DIRTBlock(dim * 2, num_heads[1], time_dim=time_dim) 
            for _ in range(num_blocks[1])])
        
        self.up1 = Upsample(dim * 2)
        self.reduce1 = nn.Conv2d(dim * 2, dim, 1)
        self.dec1 = nn.ModuleList([
            DIRTBlock(dim, num_heads[0], time_dim=time_dim) 
            for _ in range(num_blocks[0])])
        
        # Output conv
        self.conv_out = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, out_ch, 3, 1, 1)
        )
        
    def forward(self, noisy, condition, t):
        """
        Args:
            noisy: [B, 3, H, W] noisy residual
            condition: [B, 3, H, W] IR Prior output
            t: [B] timesteps
        
        Returns:
            [B, 3, H, W] predicted noise
        """
        # Timestep embedding
        t_emb = timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Concat condition
        x = torch.cat([noisy, condition], dim=1)
        x = self.conv_in(x)
        
        # Encoder
        for blk in self.enc1:
            x = blk(x, t_emb)
        skip1 = x
        x = self.down1(x)
        
        for blk in self.enc2:
            x = blk(x, t_emb)
        skip2 = x
        x = self.down2(x)
        
        for blk in self.enc3:
            x = blk(x, t_emb)
        skip3 = x
        x = self.down3(x)
        
        # Bottleneck
        for blk in self.bottleneck:
            x = blk(x, t_emb)
        
        # Decoder
        x = self.up3(x)
        x = self.reduce3(torch.cat([x, skip3], dim=1))
        for blk in self.dec3:
            x = blk(x, t_emb)
        
        x = self.up2(x)
        x = self.reduce2(torch.cat([x, skip2], dim=1))
        for blk in self.dec2:
            x = blk(x, t_emb)
        
        x = self.up1(x)
        x = self.reduce1(torch.cat([x, skip1], dim=1))
        for blk in self.dec1:
            x = blk(x, t_emb)
        
        # Output
        return self.conv_out(x)


##########################################################################
## DiffIR Main Architecture
##########################################################################

class DiffIR(nn.Module):
    """
    DiffIR: Efficient Diffusion Model for Image Restoration
    
    2-Stage 구조:
    1. CIRP (Compact IR Prior): 대략적 복원
    2. DIRT (Dynamic IR Transformer): Diffusion으로 residual 예측
    
    학습:
    - CIRP: L1 loss로 학습
    - DIRT: Diffusion loss (noise prediction)
    
    추론:
    - CIRP로 rough restoration
    - DDIM sampling으로 residual 예측
    - 최종 출력 = IR Prior + Residual
    
    Args:
        cirp_dim: CIRP 채널 수
        cirp_blocks: CIRP block 수
        dirt_dim: DIRT 채널 수
        dirt_blocks: DIRT 각 level block 수
        dirt_heads: DIRT attention heads
        num_timesteps: diffusion timesteps
        beta_start, beta_end: noise schedule
    """
    def __init__(self,
                 cirp_dim=64,
                 cirp_blocks=8,
                 dirt_dim=64,
                 dirt_blocks=[4, 6, 6, 8],
                 dirt_heads=[1, 2, 4, 8],
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        
        # Stage 1: CIRP (Compact IR Prior)
        self.cirp = CIRP(in_ch=3, dim=cirp_dim, num_blocks=cirp_blocks)
        
        # Stage 2: DIRT (Dynamic IR Transformer)
        self.dirt = DIRT(
            in_ch=6, out_ch=3, dim=dirt_dim,
            num_blocks=dirt_blocks, num_heads=dirt_heads
        )
        
        # Noise schedule (linear)
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        
    def q_sample(self, x0, t, noise=None):
        """
        Forward diffusion: add noise to clean residual
        
        q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        
        Args:
            x0: [B, C, H, W] clean residual
            t: [B] timesteps
            noise: optional pre-generated noise
        
        Returns:
            noisy: [B, C, H, W] noisy residual
            noise: [B, C, H, W] added noise
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        noisy = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        return noisy, noise
    
    def training_loss(self, degraded, clean):
        """
        Compute training loss
        
        Total loss = CIRP loss + Diffusion loss
        
        Args:
            degraded: [B, 3, H, W] input degraded image
            clean: [B, 3, H, W] target clean image
        
        Returns:
            total_loss, cirp_loss, diff_loss
        """
        B = degraded.shape[0]
        device = degraded.device
        
        # ======== Stage 1: CIRP ========
        cirp_out = self.cirp(degraded)
        cirp_loss = F.l1_loss(cirp_out, clean)
        
        # ======== Stage 2: Diffusion ========
        # Target: residual between clean and CIRP output
        residual = clean - cirp_out.detach()
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=device)
        
        # Add noise to residual
        noise = torch.randn_like(residual)
        noisy_residual, _ = self.q_sample(residual, t, noise)
        
        # Predict noise using DIRT
        noise_pred = self.dirt(noisy_residual, cirp_out.detach(), t)
        diff_loss = F.mse_loss(noise_pred, noise)
        
        # Total loss
        total_loss = cirp_loss + diff_loss
        
        return total_loss, cirp_loss, diff_loss
    
    @torch.no_grad()
    def ddim_sample(self, cirp_out, num_steps=20, eta=0.0):
        """
        DDIM sampling for fast inference
        
        DDIM: Denoising Diffusion Implicit Models
        - Deterministic sampling (eta=0)
        - 50~100 steps로도 좋은 품질
        
        Args:
            cirp_out: [B, 3, H, W] CIRP output (condition)
            num_steps: sampling steps (기본 20)
            eta: stochasticity (0 = deterministic)
        
        Returns:
            [B, 3, H, W] predicted residual
        """
        device = cirp_out.device
        B = cirp_out.shape[0]
        
        # Start from pure noise
        x = torch.randn_like(cirp_out)
        
        # DDIM timesteps (evenly spaced)
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.dirt(x, cirp_out, t_batch)
            
            # Get alpha values
            alpha = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=device)
            
            # Predict x0
            x0_pred = (x - torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(alpha)
            x0_pred = x0_pred.clamp(-1, 1)  # Clamp for stability
            
            # DDIM update
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha)) * torch.sqrt(1 - alpha / alpha_prev)
            dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * noise_pred
            
            if i + 1 < len(timesteps):
                noise = torch.randn_like(x) if eta > 0 else 0
                x = torch.sqrt(alpha_prev) * x0_pred + dir_xt + sigma * noise
            else:
                x = x0_pred
        
        return x
    
    def forward(self, degraded, num_inference_steps=20):
        """
        Inference
        
        Args:
            degraded: [B, 3, H, W] input degraded image
            num_inference_steps: DDIM sampling steps
        
        Returns:
            [B, 3, H, W] restored image
        """
        if self.training:
            raise RuntimeError("Use training_loss() for training")
        
        # Stage 1: CIRP (rough restoration)
        cirp_out = self.cirp(degraded)
        
        # Stage 2: Diffusion (residual prediction)
        residual = self.ddim_sample(cirp_out, num_steps=num_inference_steps)
        
        # Final output
        output = cirp_out + residual
        output = output.clamp(0, 1)
        
        return output


##########################################################################
## Model Variants
##########################################################################

def DiffIR_Small():
    """DiffIR Small (lighter, faster)"""
    return DiffIR(
        cirp_dim=48,
        cirp_blocks=4,
        dirt_dim=48,
        dirt_blocks=[2, 4, 4, 6],
        dirt_heads=[1, 2, 4, 8]
    )


def DiffIR_Base():
    """DiffIR Base (default)"""
    return DiffIR(
        cirp_dim=64,
        cirp_blocks=8,
        dirt_dim=64,
        dirt_blocks=[4, 6, 6, 8],
        dirt_heads=[1, 2, 4, 8]
    )


if __name__ == '__main__':
    # Test
    model = DiffIR_Small()
    x = torch.randn(1, 3, 64, 64)
    
    print(f"Input shape: {x.shape}")
    
    # Training mode test
    model.train()
    degraded = x
    clean = torch.randn(1, 3, 64, 64)
    total_loss, cirp_loss, diff_loss = model.training_loss(degraded, clean)
    print(f"Training - Total: {total_loss:.4f}, CIRP: {cirp_loss:.4f}, Diff: {diff_loss:.4f}")
    
    # Inference mode test
    model.eval()
    with torch.no_grad():
        y = model(x, num_inference_steps=5)
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
