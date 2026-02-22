# IDT: Image De-raining Transformer
# https://arxiv.org/abs/2112.02000
# IEEE TPAMI 2023
# Window-based and Spatial-based Dual Transformer for Image Deraining

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


def window_partition(x, window_size):
    """Partition feature map into non-overlapping windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Merge windows back to feature map"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network"""
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, 
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class WindowAttention(nn.Module):
    """Window-based Multi-head Self-Attention for local information"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SpatialAttention(nn.Module):
    """Spatial-based Multi-head Self-Attention for global/non-local information"""
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowTransformerBlock(nn.Module):
    """Window-based Transformer Block for local feature extraction"""
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor=mlp_ratio)

    def forward(self, x):
        B, C, H, W = x.shape
        
        shortcut = x
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm1(x)
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._get_attn_mask(H, W, x.device)
        else:
            shifted_x = x
            attn_mask = None
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = rearrange(x, 'b h w c -> b c h w')
        x = shortcut + self.drop_path(x)
        
        # FFN
        x = x + self.drop_path(self.ffn(self.norm2(rearrange(x, 'b c h w -> b (h w) c')).view(B, C, H, W).permute(0, 1, 2, 3)))
        
        return x

    def _get_attn_mask(self, H, W, device):
        """Calculate attention mask for SW-MSA"""
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


class SpatialTransformerBlock(nn.Module):
    """Spatial-based Transformer Block for non-local feature extraction"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, 
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor=mlp_ratio)

    def forward(self, x):
        B, C, H, W = x.shape
        
        shortcut = x
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm1(x)
        x = self.attn(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = shortcut + self.drop_path(x)
        
        # FFN
        shortcut = x
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm2(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.ffn(x)
        x = shortcut + self.drop_path(x)
        
        return x


class DualTransformerBlock(nn.Module):
    """Dual Transformer Block combining Window and Spatial attention"""
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        
        self.window_block = WindowTransformerBlock(
            dim=dim, num_heads=num_heads, window_size=window_size,
            shift_size=shift_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path
        )
        
        self.spatial_block = SpatialTransformerBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path
        )
        
        # Fusion
        self.fusion = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True)

    def forward(self, x):
        x_window = self.window_block(x)
        x_spatial = self.spatial_block(x)
        x = self.fusion(torch.cat([x_window, x_spatial], dim=1))
        return x


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)


class IDT(nn.Module):
    """
    IDT: Image De-raining Transformer
    
    A dual transformer architecture combining window-based and spatial-based
    self-attention for effective image deraining.
    
    Args:
        in_chans: Number of input channels (default: 3)
        embed_dim: Base embedding dimension (default: 48)
        depths: Number of transformer blocks at each stage (default: [4,6,6,8])
        num_heads: Number of attention heads at each stage (default: [1,2,4,8])
        window_size: Window size for local attention (default: 8)
        mlp_ratio: MLP expansion ratio (default: 4)
    """
    def __init__(self, in_chans=3, embed_dim=48, 
                 depths=[4, 6, 6, 8], num_heads=[1, 2, 4, 8],
                 window_size=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_levels = len(depths)
        
        # Input projection
        self.input_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, bias=True)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        for i in range(self.num_levels):
            dim = embed_dim * (2 ** i)
            blocks = []
            for j in range(depths[i]):
                shift_size = 0 if (j % 2 == 0) else window_size // 2
                blocks.append(
                    DualTransformerBlock(
                        dim=dim, num_heads=num_heads[i], window_size=window_size,
                        shift_size=shift_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i]) + j]
                    )
                )
            self.encoders.append(nn.Sequential(*blocks))
            
            if i < self.num_levels - 1:
                self.downsample.append(Downsample(dim, dim * 2))
        
        # Decoder
        self.decoders = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.skip_conv = nn.ModuleList()
        
        for i in range(self.num_levels - 2, -1, -1):
            dim = embed_dim * (2 ** i)
            self.upsample.append(Upsample(dim * 2, dim))
            self.skip_conv.append(nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True))
            
            blocks = []
            for j in range(depths[i]):
                shift_size = 0 if (j % 2 == 0) else window_size // 2
                blocks.append(
                    DualTransformerBlock(
                        dim=dim, num_heads=num_heads[i], window_size=window_size,
                        shift_size=shift_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i]) + j]
                    )
                )
            self.decoders.append(nn.Sequential(*blocks))
        
        # Output projection
        self.output_proj = nn.Conv2d(embed_dim, in_chans, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_image_size(self, x):
        """Pad image to be divisible by window_size * 2^(num_levels-1)"""
        _, _, h, w = x.size()
        padder_size = self.window_size * (2 ** (self.num_levels - 1))
        mod_pad_h = (padder_size - h % padder_size) % padder_size
        mod_pad_w = (padder_size - w % padder_size) % padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.check_image_size(x)
        
        # Input projection
        feat = self.input_proj(x)
        
        # Encoder
        enc_feats = []
        for i in range(self.num_levels):
            feat = self.encoders[i](feat)
            if i < self.num_levels - 1:
                enc_feats.append(feat)
                feat = self.downsample[i](feat)
        
        # Decoder
        for i in range(self.num_levels - 1):
            feat = self.upsample[i](feat)
            enc_feat = enc_feats[self.num_levels - 2 - i]
            feat = self.skip_conv[i](torch.cat([feat, enc_feat], dim=1))
            feat = self.decoders[i](feat)
        
        # Output projection
        out = self.output_proj(feat)
        
        # Residual connection
        return (x + out)[:, :, :H, :W]


def IDT_base():
    """IDT Base configuration"""
    return IDT(
        in_chans=3,
        embed_dim=48,
        depths=[4, 6, 6, 8],
        num_heads=[1, 2, 4, 8],
        window_size=8,
        mlp_ratio=4.
    )


def IDT_small():
    """IDT Small configuration - lighter version"""
    return IDT(
        in_chans=3,
        embed_dim=32,
        depths=[2, 4, 4, 6],
        num_heads=[1, 2, 4, 8],
        window_size=8,
        mlp_ratio=4.
    )


if __name__ == '__main__':
    # Test the model
    model = IDT()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
