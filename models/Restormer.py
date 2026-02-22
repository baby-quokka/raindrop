"""
Restormer: Efficient Transformer for High-Resolution Image Restoration
Paper: https://arxiv.org/abs/2111.09881 (CVPR 2022)
Authors: Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, 
         Fahad Shahbaz Khan, and Ming-Hsuan Yang

핵심 아이디어:
1. MDTA (Multi-DConv Head Transposed Self-Attention)
   - 기존 Self-Attention: Q, K가 spatial dimension에서 연산 → O(N²) 복잡도
   - MDTA: Channel dimension에서 연산 → O(C²) 복잡도로 효율적
   - Depth-wise Conv로 local context 보강

2. GDFN (Gated-Dconv Feed-Forward Network)
   - 기존 FFN: Linear → GELU → Linear
   - GDFN: Gating mechanism + Depth-wise Conv 추가
   - 더 풍부한 feature transformation

3. Progressive U-Net 구조
   - 4-level encoder-decoder with skip connections
   - 각 level에서 다른 수의 Transformer blocks
   - Refinement stage로 최종 품질 향상
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange


##########################################################################
## Layer Norm 관련 함수 및 클래스
##########################################################################

def to_3d(x):
    """
    4D 텐서를 3D로 변환 (LayerNorm 적용을 위해)
    
    Args:
        x: [B, C, H, W] 형태의 텐서
    Returns:
        [B, H*W, C] 형태의 텐서
    
    이유: PyTorch LayerNorm은 마지막 차원에 대해 정규화하므로
          채널을 마지막으로 이동시켜야 함
    """
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    """
    3D 텐서를 다시 4D로 변환
    
    Args:
        x: [B, H*W, C] 형태의 텐서
        h, w: 원래 height, width
    Returns:
        [B, C, H, W] 형태의 텐서
    """
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    """
    Bias가 없는 LayerNorm
    
    수식: y = (x - mean) / sqrt(var + eps) * weight
    
    Bias를 제거한 이유:
    - 파라미터 수 감소
    - 일부 task에서 bias 없이도 충분한 성능
    - Restormer 논문에서 실험적으로 검증
    """
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        # normalized_shape: 정규화할 차원의 크기 (예: [3, 48, 48])
        # 정수인 경우 tuple로 변환
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        # normalized_shape를 torch.Size 타입으로 변환
        normalized_shape = torch.Size(normalized_shape)
        # normalized_shape의 길이가 1이어야 함
        # (assert: 조건이 참이 아니면 오류 발생)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # variance만 사용 (mean 빼지 않음 - BiasFree 특성)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """
    일반적인 LayerNorm (bias 포함)
    
    수식: y = (x - mean) / sqrt(var + eps) * weight + bias
    
    Restormer 기본 설정에서 사용됨
    """
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # x.mean(-1, keepdim=True): 마지막 차원(-1)에 대해 평균 계산
        # x.var(-1, keepdim=True, unbiased=False): 마지막 차원(-1)에 대해 분산 계산
        # 분산 계산 시 unbiased=False 옵션 사용 (비편향 분산 계산)
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """
    LayerNorm Wrapper
    
    4D 텐서 [B, C, H, W]를 받아서:
    1. 3D로 변환 [B, H*W, C]
    2. LayerNorm 적용 (채널 차원에 대해)
    3. 다시 4D로 변환 [B, C, H, W]
    
    Args:
        dim: 채널 수
        LayerNorm_type: 'BiasFree' 또는 'WithBias'
    """
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## GDFN (Gated-Dconv Feed-Forward Network)
##########################################################################

class FeedForward(nn.Module):
    """
    Gated-Dconv Feed-Forward Network (GDFN)
    
    기존 FFN과의 차이점:
    1. Gating Mechanism: x1 * GELU(x2) 형태로 정보 흐름 제어
    2. Depth-wise Conv: 3x3 DWConv로 local spatial context 추가
    
    구조:
        Input [B, C, H, W]
          ↓
        1x1 Conv (C → 2*hidden)     # 채널 확장
          ↓
        3x3 DWConv (그룹=2*hidden)   # Local context
          ↓
        Split → x1, x2
          ↓
        GELU(x1) * x2               # Gating
          ↓
        1x1 Conv (hidden → C)       # 채널 복원
          ↓
        Output [B, C, H, W]
    
    Args:
        dim: 입력 채널 수
        ffn_expansion_factor: hidden 채널 확장 비율 (기본 2.66)
        bias: bias 사용 여부
    """
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        # 1x1 Conv: 채널을 2배 확장 (gating을 위해)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # 3x3 Depth-wise Conv: 각 채널 독립적으로 spatial 정보 처리
        # groups=hidden_features*2 → 완전한 depth-wise convolution
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 
                                kernel_size=3, stride=1, padding=1, 
                                groups=hidden_features * 2, bias=bias)

        # 1x1 Conv: 채널 복원
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # 채널 방향으로 반으로 분할
        x = F.gelu(x1) * x2  # Gating: 활성화된 x1이 x2의 정보 흐름 제어
        x = self.project_out(x)
        return x


##########################################################################
## MDTA (Multi-DConv Head Transposed Self-Attention)
##########################################################################

class Attention(nn.Module):
    """
    Multi-DConv Head Transposed Self-Attention (MDTA)
    
    핵심 아이디어 - Transposed Attention:
    - 기존 Self-Attention: Attention(Q, K) = softmax(Q @ K^T / sqrt(d))
      - Q, K: [B, N, C] where N = H*W (spatial tokens)
      - 복잡도: O(N²) = O((H*W)²) → 고해상도에서 매우 비효율적
    
    - Transposed Attention: Q, K를 transpose하여 채널 간 attention 계산
      - Q, K: [B, C, N] → Attention은 [C, C] 크기
      - 복잡도: O(C²) → 채널 수는 고정이므로 해상도에 무관
    
    추가 개선:
    1. Multi-head: 여러 attention head로 다양한 패턴 학습
    2. Depth-wise Conv: Q, K, V 생성 시 local context 추가
    3. L2 Normalization: Q, K에 적용하여 학습 안정화
    4. Learnable Temperature: head별로 학습 가능한 스케일링
    
    구조:
        Input [B, C, H, W]
          ↓
        1x1 Conv (C → 3C)          # Q, K, V 생성
          ↓
        3x3 DWConv                  # Local context 추가
          ↓
        Split → Q, K, V
          ↓
        Reshape to [B, heads, C/heads, H*W]
          ↓
        L2 Normalize Q, K
          ↓
        Attention = softmax(Q @ K^T * temperature)  # [C/heads, C/heads]
          ↓
        Out = Attention @ V
          ↓
        Reshape to [B, C, H, W]
          ↓
        1x1 Conv projection
          ↓
        Output [B, C, H, W]
    
    Args:
        dim: 입력 채널 수
        num_heads: attention head 수
        bias: bias 사용 여부
    """
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        
        # Learnable temperature (head별로 다른 스케일링)
        # 기존 sqrt(d)로 나누는 대신, 학습 가능한 파라미터 사용
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Q, K, V를 한번에 생성하는 1x1 Conv
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        
        # 3x3 Depth-wise Conv: Q, K, V에 local spatial context 추가
        # 이것이 "Multi-DConv"의 핵심 - 순수 attention 대비 local 정보 보강
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, 
                                     padding=1, groups=dim * 3, bias=bias)
        
        # Output projection
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        # Q, K, V 생성: 1x1 Conv → 3x3 DWConv
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Multi-head를 위한 reshape
        # [B, C, H, W] → [B, heads, C/heads, H*W]
        # 주의: 일반적인 attention과 달리 (C/heads, H*W) 순서
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # L2 Normalization: 학습 안정화
        # Q, K를 정규화하면 dot product 값이 bounded → softmax 안정
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Transposed Attention 계산
        # Q: [B, heads, C/heads, N], K: [B, heads, C/heads, N]
        # Q @ K^T: [B, heads, C/heads, C/heads] ← 채널 간 attention!
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # Clamp attention scores to prevent overflow in softmax
        attn = attn.clamp(-50, 50)
        attn = attn.softmax(dim=-1)

        # Attention 적용
        # attn @ V: [B, heads, C/heads, N]
        out = (attn @ v)

        # Reshape back: [B, heads, C/heads, H*W] → [B, C, H, W]
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', 
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Transformer Block
##########################################################################

class TransformerBlock(nn.Module):
    """
    Restormer Transformer Block
    
    구조 (Pre-norm 방식):
        Input
          ↓
        LayerNorm → MDTA → + (residual)
          ↓
        LayerNorm → GDFN → + (residual)
          ↓
        Output
    
    Pre-norm vs Post-norm:
    - Post-norm: x + LayerNorm(Attention(x))
    - Pre-norm: x + Attention(LayerNorm(x))  ← Restormer 사용
    - Pre-norm이 학습 안정성이 더 좋음 (특히 deep network에서)
    
    Args:
        dim: 채널 수
        num_heads: attention head 수
        ffn_expansion_factor: GDFN 확장 비율
        bias: bias 사용 여부
        LayerNorm_type: 'BiasFree' 또는 'WithBias'
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # Pre-norm + Residual
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## Patch Embedding
##########################################################################

class OverlapPatchEmbed(nn.Module):
    """
    Overlapping Patch Embedding
    
    ViT의 non-overlapping patch embedding과 달리, 
    3x3 Conv로 overlapping하게 patch 생성
    
    장점:
    - Patch boundary artifacts 감소
    - Spatial continuity 보존
    - 더 자연스러운 feature extraction
    
    구조:
        Input [B, 3, H, W]
          ↓
        3x3 Conv (stride=1, padding=1)  # 해상도 유지
          ↓
        Output [B, embed_dim, H, W]
    
    Args:
        in_c: 입력 채널 (RGB=3)
        embed_dim: 출력 채널 (기본 48)
        bias: bias 사용 여부
    """
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Downsample / Upsample
##########################################################################

class Downsample(nn.Module):
    """
    Downsampling Module (해상도 1/2로 축소)
    
    구조:
        Input [B, C, H, W]
          ↓
        3x3 Conv (C → C/2)      # 채널 절반으로
          ↓
        PixelUnshuffle(2)       # [B, C/2, H, W] → [B, 2C, H/2, W/2]
          ↓
        Output [B, 2C, H/2, W/2]
    
    PixelUnshuffle:
    - PixelShuffle의 역연산
    - 공간 정보를 채널로 재배치
    - [B, C, 2H, 2W] → [B, 4C, H, W]
    
    결과적으로: 채널 2배, 해상도 1/2
    """
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)  # 해상도 1/2, 채널 4배 → 총 채널 2배
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """
    Upsampling Module (해상도 2배로 확대)
    
    구조:
        Input [B, C, H, W]
          ↓
        3x3 Conv (C → 2C)       # 채널 2배로
          ↓
        PixelShuffle(2)         # [B, 2C, H, W] → [B, C/2, 2H, 2W]
          ↓
        Output [B, C/2, 2H, 2W]
    
    PixelShuffle (Sub-pixel Convolution):
    - 채널의 정보를 공간으로 재배치
    - [B, 4C, H, W] → [B, C, 2H, 2W]
    - Deconvolution 대비 checkerboard artifact 없음
    
    결과적으로: 채널 1/2, 해상도 2배
    """
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)  # 해상도 2배, 채널 1/4 → 총 채널 1/2
        )

    def forward(self, x):
        return self.body(x)


##########################################################################
## Restormer Main Architecture
##########################################################################

class Restormer(nn.Module):
    """
    Restormer: Efficient Transformer for High-Resolution Image Restoration
    
    전체 구조 (4-Level U-Net):
    
    Input [B, 3, H, W]
      ↓
    OverlapPatchEmbed [B, 48, H, W]
      ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ ENCODER                                                     │
    │                                                             │
    │ Level 1: [B, 48, H, W]      - 4 Transformer Blocks          │
    │     ↓ Downsample                                            │
    │ Level 2: [B, 96, H/2, W/2]  - 6 Transformer Blocks          │
    │     ↓ Downsample                                            │
    │ Level 3: [B, 192, H/4, W/4] - 6 Transformer Blocks          │
    │     ↓ Downsample                                            │
    │ Level 4 (Latent): [B, 384, H/8, W/8] - 8 Transformer Blocks │
    └─────────────────────────────────────────────────────────────┘
      ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ DECODER (with Skip Connections)                             │
    │                                                             │
    │     ↑ Upsample + Concat(skip) + 1x1 Conv                    │
    │ Level 3: [B, 192, H/4, W/4] - 6 Transformer Blocks          │
    │     ↑ Upsample + Concat(skip) + 1x1 Conv                    │
    │ Level 2: [B, 96, H/2, W/2]  - 6 Transformer Blocks          │
    │     ↑ Upsample + Concat(skip)                               │
    │ Level 1: [B, 96, H, W]      - 4 Transformer Blocks          │
    └─────────────────────────────────────────────────────────────┘
      ↓
    Refinement: [B, 96, H, W] - 4 Transformer Blocks
      ↓
    3x3 Conv [B, 3, H, W] + Input (Global Residual)
      ↓
    Output [B, 3, H, W]
    
    채널 변화:
    - Level 1: 48 (dim)
    - Level 2: 96 (dim * 2)
    - Level 3: 192 (dim * 4)
    - Level 4: 384 (dim * 8)
    
    Args:
        inp_channels: 입력 채널 (기본 3, RGB)
        out_channels: 출력 채널 (기본 3, RGB)
        dim: base 채널 수 (기본 48)
        num_blocks: 각 level의 Transformer block 수 [4, 6, 6, 8]
        num_refinement_blocks: refinement stage block 수 (기본 4)
        heads: 각 level의 attention head 수 [1, 2, 4, 8]
        ffn_expansion_factor: GDFN 확장 비율 (기본 2.66)
        bias: bias 사용 여부 (기본 False)
        LayerNorm_type: LayerNorm 종류 (기본 'WithBias')
        dual_pixel_task: dual-pixel defocus deblurring용 (기본 False)
    """
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False,
                 use_multi_scale=False):
        
        super(Restormer, self).__init__()
        
        self.use_multi_scale = use_multi_scale

        # -------- Patch Embedding --------
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # -------- Encoder --------
        # Level 1: [B, 48, H, W]
        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[0], 
                               ffn_expansion_factor=ffn_expansion_factor, 
                               bias=bias, LayerNorm_type=LayerNorm_type) 
              for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  # 48 → 96, H → H/2

        # Level 2: [B, 96, H/2, W/2]
        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
              for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  # 96 → 192, H/2 → H/4

        # Level 3: [B, 192, H/4, W/4]
        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
              for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  # 192 → 384, H/4 → H/8

        # Level 4 (Latent/Bottleneck): [B, 384, H/8, W/8]
        self.latent = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3],
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
              for i in range(num_blocks[3])])

        # -------- Decoder --------
        self.up4_3 = Upsample(int(dim * 2 ** 3))  # 384 → 192, H/8 → H/4
        # Skip connection 후 채널 감소: 192 + 192 = 384 → 192
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), 
                                            kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
              for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  # 192 → 96, H/4 → H/2
        # Skip connection 후 채널 감소: 96 + 96 = 192 → 96
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1),
                                            kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
              for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  # 96 → 48, H/2 → H
        # Level 1에서는 채널 감소 없이 concat: 48 + 48 = 96
        self.decoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0],
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
              for i in range(num_blocks[0])])

        # -------- Refinement Stage --------
        # 추가적인 품질 향상을 위한 후처리 블록
        self.refinement = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0],
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
              for i in range(num_refinement_blocks)])

        # -------- Dual-Pixel Task (Optional) --------
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)

        # -------- Output Projection --------
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, 
                                kernel_size=3, stride=1, padding=1, bias=bias)
        
        # -------- Multi-scale Output Projections (for multi-scale supervision) --------
        if self.use_multi_scale:
            # x/4 scale output (from decoder_level3)
            self.output_level3 = nn.Conv2d(int(dim * 2 ** 2), out_channels,
                                           kernel_size=3, stride=1, padding=1, bias=bias)
            # x/2 scale output (from decoder_level2)
            self.output_level2 = nn.Conv2d(int(dim * 2 ** 1), out_channels,
                                           kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        """
        Forward pass
        
        Args:
            inp_img: [B, 3, H, W] 입력 이미지
        
        Returns:
            [B, 3, H, W] 복원된 이미지
        """
        # -------- Patch Embedding --------
        inp_enc_level1 = self.patch_embed(inp_img)  # [B, 48, H, W]

        # -------- Encoder --------
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # [B, 48, H, W]

        inp_enc_level2 = self.down1_2(out_enc_level1)  # [B, 96, H/2, W/2]
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)  # [B, 192, H/4, W/4]
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)  # [B, 384, H/8, W/8]
        latent = self.latent(inp_enc_level4)  # Bottleneck

        # -------- Decoder with Skip Connections --------
        inp_dec_level3 = self.up4_3(latent)  # [B, 192, H/4, W/4]
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)  # Skip connection
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)  # 채널 감소
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        # Multi-scale output: x/4 scale
        if self.use_multi_scale:
            # 입력을 x/4로 다운샘플링하여 residual connection
            inp_img_level3 = F.interpolate(inp_img, size=out_dec_level3.shape[2:], mode='bilinear', align_corners=False)
            out_level3 = self.output_level3(out_dec_level3) + inp_img_level3
        else:
            out_level3 = None

        inp_dec_level2 = self.up3_2(out_dec_level3)  # [B, 96, H/2, W/2]
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)  # Skip connection
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)  # 채널 감소
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        # Multi-scale output: x/2 scale
        if self.use_multi_scale:
            # 입력을 x/2로 다운샘플링하여 residual connection
            inp_img_level2 = F.interpolate(inp_img, size=out_dec_level2.shape[2:], mode='bilinear', align_corners=False)
            out_level2 = self.output_level2(out_dec_level2) + inp_img_level2
        else:
            out_level2 = None

        inp_dec_level1 = self.up2_1(out_dec_level2)  # [B, 48, H, W]
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)  # [B, 96, H, W]
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # -------- Refinement --------
        out_dec_level1 = self.refinement(out_dec_level1)

        # -------- Output --------
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            # Global Residual Learning: output + input
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        # Multi-scale 출력 반환
        if self.use_multi_scale:
            return [out_level3, out_level2, out_dec_level1]  # [x/4, x/2, x]
        else:
            return out_dec_level1


##########################################################################
## Model Variants
##########################################################################

def Restormer_Small():
    """Restormer Small variant (lighter)"""
    return Restormer(
        dim=32,
        num_blocks=[2, 4, 4, 6],
        num_refinement_blocks=2,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66
    )


def Restormer_Base():
    """
    Restormer Base variant (default)
    
    Multi-scale 활성화 방법:
    - 옵션 1: Restormer_Base_MultiScale() 사용 (권장)
    - 옵션 2: 아래 use_multi_scale=True로 변경
    """
    return Restormer(
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        use_multi_scale=False  # True로 변경하면 Multi-scale 활성화
    )


def Restormer_Base_MultiScale():
    """Restormer Base variant with multi-scale supervision"""
    return Restormer(
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        use_multi_scale=True
    )


if __name__ == '__main__':
    # Test
    model = Restormer()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
