"""
MambaIR: A Simple Baseline for Image Restoration with State-Space Model
Paper: https://arxiv.org/abs/2402.15648

순수 PyTorch 구현 (mamba-ssm 불필요)
- Selective State Space Model (S6) 직접 구현
- 4방향 스캔 (Visual State Space)
- CUDA 최적화 없이 동작 (약간 느림)

핵심 아이디어:
1. SSM (State Space Model): 연속 시스템을 이산화하여 시퀀스 모델링
   - x'(t) = Ax(t) + Bu(t)
   - y(t) = Cx(t) + Du(t)
   
2. Selective SSM: 입력에 따라 B, C, Δ를 동적으로 생성
   - 기존 SSM: 고정된 파라미터
   - Selective SSM: 입력에 dependent한 파라미터 → content-aware
   
3. Visual State Space: 2D 이미지를 1D 시퀀스로 변환하여 처리
   - 4방향 스캔으로 spatial context 보존
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


##########################################################################
## Selective SSM (순수 PyTorch 구현)
##########################################################################

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) - Mamba의 핵심
    
    기존 SSM과의 차이:
    - 기존: A, B, C, D가 학습 가능하지만 고정된 파라미터
    - Selective: B, C, Δ가 입력에 따라 동적으로 생성 (input-dependent)
    
    수식:
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C * h_t + D * x_t
        
        where:
        A_bar = exp(Δ * A)  # Discretized A
        B_bar = Δ * B        # Discretized B
    
    Args:
        d_model: 입력 차원
        d_state: SSM state 차원 (N in paper, 기본 16)
        d_conv: Local convolution 커널 크기
        expand: 내부 차원 확장 비율
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # Input projection: x → (x, z) for gating
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Local convolution (depth-wise)
        # Mamba에서는 이 conv가 local context를 제공
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        
        # Selective parameters projection
        # x → (B, C, Δ) : input-dependent SSM parameters
        # dt_rank는 Δ projection의 bottleneck 차원
        self.dt_rank = math.ceil(d_model / 16)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # Δ (delta) projection: dt_rank → d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # SSM Parameters
        # A: State transition matrix (d_inner, d_state)
        # 초기화: A를 음수로 (안정적인 dynamics를 위해)
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), 
                   'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))  # log space에서 학습
        
        # D: Skip connection (residual)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: [B, L, D] 입력 시퀀스
        Returns:
            [B, L, D] 출력 시퀀스
        """
        batch, seq_len, _ = x.shape
        
        # ======== Input Projection ========
        # [B, L, D] → [B, L, 2*d_inner]
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # x, z: [B, L, d_inner]
        
        # ======== Local Convolution ========
        # [B, L, d_inner] → [B, d_inner, L] for conv1d
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]  # Causal: 미래 정보 사용 안함
        x = x.transpose(1, 2)  # [B, L, d_inner]
        
        x = F.silu(x)  # Activation
        
        # ======== Selective SSM ========
        y = self.ssm(x)
        
        # ======== Gating ========
        # z가 gate 역할: 어떤 정보를 출력할지 제어
        y = y * F.silu(z)
        
        # ======== Output Projection ========
        output = self.out_proj(y)
        
        return output
    
    def ssm(self, x):
        """
        Selective State Space Model 연산
        
        Args:
            x: [B, L, d_inner] convolution 후 입력
        Returns:
            [B, L, d_inner] SSM 출력
        """
        batch, seq_len, d_inner = x.shape
        
        # ======== A matrix (state transition) ========
        # A_log에서 A 복원 (음수로 유지하여 안정성 보장)
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # ======== Selective Parameters (B, C, Δ) ========
        # x에서 B, C, Δ를 동적으로 생성
        x_proj = self.x_proj(x)  # [B, L, dt_rank + 2*d_state]
        
        # Split into delta, B, C
        delta, B, C = x_proj.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        # Δ projection + softplus (양수 보장)
        delta = F.softplus(self.dt_proj(delta))  # [B, L, d_inner]
        
        # ======== Discretization ========
        # 연속 SSM → 이산 SSM 변환
        # A_bar = exp(Δ * A)
        # B_bar = Δ * B
        
        # ======== Selective Scan ========
        y = self.selective_scan(x, delta, A, B, C)
        
        # ======== Skip Connection ========
        y = y + x * self.D
        
        return y
    
    def selective_scan(self, x, delta, A, B, C):
        """
        Selective Scan 연산 (순수 PyTorch, loop 버전)
        
        이 부분이 mamba-ssm CUDA 커널이 최적화하는 부분
        순수 PyTorch로는 느리지만 동일한 결과
        
        Args:
            x: [B, L, d_inner] 입력
            delta: [B, L, d_inner] discretization step
            A: [d_inner, d_state] state transition
            B: [B, L, d_state] input-dependent B
            C: [B, L, d_state] input-dependent C
        
        Returns:
            [B, L, d_inner] 출력
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Discretize A: A_bar = exp(delta * A)
        # delta: [B, L, d_inner], A: [d_inner, d_state]
        # → deltaA: [B, L, d_inner, d_state]
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        
        # Discretize B: B_bar = delta * B
        # delta: [B, L, d_inner], B: [B, L, d_state]
        # → deltaB: [B, L, d_inner, d_state]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        
        # x를 확장: [B, L, d_inner, 1]
        x_expanded = x.unsqueeze(-1)
        
        # ======== Recurrence (Sequential Scan) ========
        # h_t = A_bar * h_{t-1} + B_bar * x_t
        # y_t = C * h_t
        
        # 초기 hidden state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        for t in range(seq_len):
            # State update: h = A_bar * h + B_bar * x
            h = deltaA[:, t] * h + deltaB[:, t] * x_expanded[:, t]
            
            # Output: y = C * h (sum over state dimension)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # [B, d_inner]
            outputs.append(y_t)
        
        # Stack outputs: [B, L, d_inner]
        y = torch.stack(outputs, dim=1)
        
        return y


##########################################################################
## Visual State Space Block (VSS Block)
##########################################################################

class VSSBlock(nn.Module):
    """
    Visual State Space Block
    
    2D 이미지를 1D 시퀀스로 변환하여 SSM 적용
    4방향 스캔으로 spatial context 보존:
    - Forward (좌→우, 상→하)
    - Backward (우→좌, 하→상)
    - Forward Transpose (상→하, 좌→우)
    - Backward Transpose (하→상, 우→좌)
    
    구조:
        Input [B, C, H, W]
          ↓
        LayerNorm
          ↓
        4-direction Scan (병렬)
          ↓
        Merge + Conv
          ↓
        + (residual)
          ↓
        Output [B, C, H, W]
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.):
        super().__init__()
        
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        
        # 4방향 SSM
        self.ssm_forward = SelectiveSSM(dim, d_state, d_conv, expand)
        self.ssm_backward = SelectiveSSM(dim, d_state, d_conv, expand)
        self.ssm_forward_t = SelectiveSSM(dim, d_state, d_conv, expand)  # Transpose
        self.ssm_backward_t = SelectiveSSM(dim, d_state, d_conv, expand)  # Transpose
        
        # Merge conv
        self.merge = nn.Conv2d(dim * 4, dim, 1)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape
        residual = x
        
        # [B, C, H, W] → [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        
        # ======== 4-direction Scan ========
        # Direction 1: Forward (row-major, 좌상→우하)
        x_flat_fwd = x.reshape(B, H * W, C)
        y_fwd = self.ssm_forward(x_flat_fwd)
        
        # Direction 2: Backward (우하→좌상)
        x_flat_bwd = torch.flip(x_flat_fwd, dims=[1])
        y_bwd = self.ssm_backward(x_flat_bwd)
        y_bwd = torch.flip(y_bwd, dims=[1])
        
        # Direction 3: Forward Transpose (column-major, 좌상→우하)
        x_t = x.permute(0, 2, 1, 3).reshape(B, H * W, C)  # [B, W*H, C]
        y_fwd_t = self.ssm_forward_t(x_t)
        y_fwd_t = y_fwd_t.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, H * W, C)
        
        # Direction 4: Backward Transpose
        x_t_bwd = torch.flip(x_t, dims=[1])
        y_bwd_t = self.ssm_backward_t(x_t_bwd)
        y_bwd_t = torch.flip(y_bwd_t, dims=[1])
        y_bwd_t = y_bwd_t.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, H * W, C)
        
        # ======== Merge ========
        # [B, H*W, C] → [B, C, H, W]
        y_fwd = y_fwd.reshape(B, H, W, C).permute(0, 3, 1, 2)
        y_bwd = y_bwd.reshape(B, H, W, C).permute(0, 3, 1, 2)
        y_fwd_t = y_fwd_t.reshape(B, H, W, C).permute(0, 3, 1, 2)
        y_bwd_t = y_bwd_t.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Concatenate and merge
        y = torch.cat([y_fwd, y_bwd, y_fwd_t, y_bwd_t], dim=1)  # [B, 4C, H, W]
        y = self.merge(y)  # [B, C, H, W]
        
        # Residual
        x = residual + y
        
        # ======== FFN ========
        # [B, C, H, W] → [B, H, W, C]
        x_ffn = x.permute(0, 2, 3, 1)
        x_ffn = x_ffn + self.ffn(x_ffn)
        x = x_ffn.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return x


##########################################################################
## Channel Attention
##########################################################################

class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    
    각 채널의 중요도를 학습하여 가중치 적용
    
    구조:
        Input [B, C, H, W]
          ↓
        Global Average Pooling → [B, C, 1, 1]
          ↓
        FC → ReLU → FC → Sigmoid
          ↓
        * Input (channel-wise multiplication)
          ↓
        Output [B, C, H, W]
    """
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        # Global Average Pooling
        y = self.avg_pool(x).view(B, C)
        # FC layers
        y = self.fc(y).view(B, C, 1, 1)
        # Channel-wise multiplication
        return x * y.expand_as(x)


##########################################################################
## MambaIR Block
##########################################################################

class MambaIRBlock(nn.Module):
    """
    MambaIR Basic Block
    
    구조:
        Input
          ↓
        VSS Block (Selective SSM)
          ↓
        Conv 3x3
          ↓
        Channel Attention
          ↓
        + (residual)
          ↓
        Output
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        
        self.vss = VSSBlock(dim, d_state, d_conv, expand)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.ca = ChannelAttention(dim)
        
    def forward(self, x):
        residual = x
        
        # VSS Block
        x = self.vss(x)
        
        # Conv + CA
        x = self.conv(x)
        x = self.ca(x)
        
        # Residual
        return x + residual


##########################################################################
## Downsample / Upsample
##########################################################################

class Downsample(nn.Module):
    """Downsample: 해상도 1/2, 채널 2배"""
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, 1, 1),
            nn.PixelUnshuffle(2)
        )
    
    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """Upsample: 해상도 2배, 채널 1/2"""
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.PixelShuffle(2)
        )
    
    def forward(self, x):
        return self.body(x)


##########################################################################
## MambaIR Main Architecture
##########################################################################

class MambaIR(nn.Module):
    """
    MambaIR: Image Restoration with State-Space Model
    
    구조 (U-Net style):
        Input [B, 3, H, W]
          ↓
        Shallow Feature Extraction
          ↓
        Encoder (VSS Blocks + Downsample)
          ↓
        Bottleneck
          ↓
        Decoder (VSS Blocks + Upsample + Skip)
          ↓
        Reconstruction
          ↓
        + Input (Global Residual)
          ↓
        Output [B, 3, H, W]
    
    Args:
        in_ch: 입력 채널 (기본 3)
        dim: base 채널 수 (기본 48)
        num_blocks: 각 stage의 block 수 (기본 [4, 6, 6, 8])
        d_state: SSM state 차원 (기본 16)
        d_conv: SSM conv 커널 크기 (기본 4)
        expand: SSM 내부 차원 확장 (기본 2)
    """
    def __init__(self,
                 in_ch=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 d_state=16,
                 d_conv=4,
                 expand=2):
        super().__init__()
        
        # -------- Shallow Feature Extraction --------
        self.conv_first = nn.Conv2d(in_ch, dim, 3, 1, 1)
        
        # -------- Encoder --------
        self.encoder1 = nn.Sequential(
            *[MambaIRBlock(dim, d_state, d_conv, expand) 
              for _ in range(num_blocks[0])])
        self.down1 = Downsample(dim)
        
        self.encoder2 = nn.Sequential(
            *[MambaIRBlock(dim * 2, d_state, d_conv, expand) 
              for _ in range(num_blocks[1])])
        self.down2 = Downsample(dim * 2)
        
        self.encoder3 = nn.Sequential(
            *[MambaIRBlock(dim * 4, d_state, d_conv, expand) 
              for _ in range(num_blocks[2])])
        self.down3 = Downsample(dim * 4)
        
        # -------- Bottleneck --------
        self.bottleneck = nn.Sequential(
            *[MambaIRBlock(dim * 8, d_state, d_conv, expand) 
              for _ in range(num_blocks[3])])
        
        # -------- Decoder --------
        self.up3 = Upsample(dim * 8)
        self.reduce3 = nn.Conv2d(dim * 8, dim * 4, 1)
        self.decoder3 = nn.Sequential(
            *[MambaIRBlock(dim * 4, d_state, d_conv, expand) 
              for _ in range(num_blocks[2])])
        
        self.up2 = Upsample(dim * 4)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, 1)
        self.decoder2 = nn.Sequential(
            *[MambaIRBlock(dim * 2, d_state, d_conv, expand) 
              for _ in range(num_blocks[1])])
        
        self.up1 = Upsample(dim * 2)
        self.reduce1 = nn.Conv2d(dim * 2, dim, 1)
        self.decoder1 = nn.Sequential(
            *[MambaIRBlock(dim, d_state, d_conv, expand) 
              for _ in range(num_blocks[0])])
        
        # -------- Reconstruction --------
        self.conv_last = nn.Conv2d(dim, in_ch, 3, 1, 1)
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 입력 이미지
        Returns:
            [B, 3, H, W] 복원된 이미지
        """
        # Global residual
        inp = x
        
        # Shallow feature
        x = self.conv_first(x)
        
        # Encoder
        enc1 = self.encoder1(x)
        x = self.down1(enc1)
        
        enc2 = self.encoder2(x)
        x = self.down2(enc2)
        
        enc3 = self.encoder3(x)
        x = self.down3(enc3)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.up3(x)
        x = self.reduce3(torch.cat([x, enc3], dim=1))
        x = self.decoder3(x)
        
        x = self.up2(x)
        x = self.reduce2(torch.cat([x, enc2], dim=1))
        x = self.decoder2(x)
        
        x = self.up1(x)
        x = self.reduce1(torch.cat([x, enc1], dim=1))
        x = self.decoder1(x)
        
        # Reconstruction + Global residual
        x = self.conv_last(x) + inp
        
        return x


##########################################################################
## Model Variants
##########################################################################

def MambaIR_Tiny():
    """MambaIR Tiny - 테스트용 최소 버전 (Pure PyTorch SSM용)"""
    return MambaIR(
        dim=16,
        num_blocks=[1, 1, 1, 1],
        d_state=4,
        d_conv=2,
        expand=1
    )


def MambaIR_Small():
    """MambaIR Small (lighter)"""
    return MambaIR(
        dim=32,
        num_blocks=[2, 4, 4, 6],
        d_state=8
    )


def MambaIR_Base():
    """MambaIR Base (default)"""
    return MambaIR(
        dim=48,
        num_blocks=[4, 6, 6, 8],
        d_state=16
    )


if __name__ == '__main__':
    # Test
    model = MambaIR_Small()  # Small for testing
    x = torch.randn(1, 3, 64, 64)  # Small input for speed
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
