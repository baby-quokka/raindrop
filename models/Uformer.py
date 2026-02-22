# Uformer: A General U-Shaped Transformer for Image Restoration
# https://arxiv.org/abs/2106.03106
# Based on: https://github.com/ZhendongWang6/Uformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


def window_partition(x, win_size):
    """Partition into windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)
    return windows


def window_reverse(windows, win_size, H, W):
    """Reverse window partition"""
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class LeFF(nn.Module):
    """Locally-enhanced Feed-Forward Network"""
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_layer()
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer()
        )
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        
        x = self.linear1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=hh, w=hh)
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=hh, w=hh)
        x = self.linear2(x)
        
        return x


class WindowAttention(nn.Module):
    """Window-based Multi-head Self-Attention"""
    def __init__(self, dim, win_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads)
        )
        
        coords_h = torch.arange(self.win_size[0])
        coords_w = torch.arange(self.win_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.win_size[0] - 1
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LeWinTransformerBlock(nn.Module):
    """LeWin Transformer Block with window-based attention"""
    def __init__(self, dim, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=(win_size, win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, mask=None):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._get_attn_mask(H, W, x.device)
        else:
            shifted_x = x
            attn_mask = None
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.win_size)
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

    def _get_attn_mask(self, H, W, device):
        """Calculate attention mask for SW-MSA"""
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.win_size),
                    slice(-self.win_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.win_size),
                    slice(-self.win_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.win_size)
        mask_windows = mask_windows.view(-1, self.win_size * self.win_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


class BasicUformerLayer(nn.Module):
    """Basic Uformer layer consisting of multiple LeWin Transformer blocks"""
    def __init__(self, dim, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, shift_flag=True):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        self.blocks = nn.ModuleList([
            LeWinTransformerBlock(
                dim=dim, num_heads=num_heads, win_size=win_size,
                shift_size=0 if (i % 2 == 0) or not shift_flag else win_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

    def forward(self, x, mask=None):
        for blk in self.blocks:
            x = blk(x, mask)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()
        return out


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()
        return out


class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()
        return x


class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1):
        super().__init__()
        self.proj = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        return x


class Uformer(nn.Module):
    """
    Uformer: A General U-Shaped Transformer for Image Restoration
    
    A U-shaped architecture with LeWin Transformer blocks for efficient
    image restoration.
    
    Args:
        img_size: Input image size (default: 256)
        in_chans: Number of input channels (default: 3)
        embed_dim: Base embedding dimension (default: 32)
        depths: Number of blocks at each stage (default: [2,2,2,2,2,2,2,2,2])
        num_heads: Number of attention heads at each stage
        win_size: Window size for local attention (default: 8)
        mlp_ratio: MLP hidden dim ratio (default: 4)
    """
    def __init__(self, img_size=256, in_chans=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], 
                 num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, shift_flag=True):
        super().__init__()
        
        self.num_enc_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.win_size = win_size
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Input/Output projections
        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim)
        self.output_proj = OutputProj(in_channel=2*embed_dim, out_channel=in_chans)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(
            dim=embed_dim, depth=depths[0], num_heads=num_heads[0],
            win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[:depths[0]], norm_layer=norm_layer, shift_flag=shift_flag
        )
        self.dowsample_0 = Downsample(embed_dim, embed_dim*2)
        
        self.encoderlayer_1 = BasicUformerLayer(
            dim=embed_dim*2, depth=depths[1], num_heads=num_heads[1],
            win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:1]):sum(depths[:2])], norm_layer=norm_layer, shift_flag=shift_flag
        )
        self.dowsample_1 = Downsample(embed_dim*2, embed_dim*4)
        
        self.encoderlayer_2 = BasicUformerLayer(
            dim=embed_dim*4, depth=depths[2], num_heads=num_heads[2],
            win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:2]):sum(depths[:3])], norm_layer=norm_layer, shift_flag=shift_flag
        )
        self.dowsample_2 = Downsample(embed_dim*4, embed_dim*8)
        
        self.encoderlayer_3 = BasicUformerLayer(
            dim=embed_dim*8, depth=depths[3], num_heads=num_heads[3],
            win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:3]):sum(depths[:4])], norm_layer=norm_layer, shift_flag=shift_flag
        )
        self.dowsample_3 = Downsample(embed_dim*8, embed_dim*16)
        
        # Bottleneck
        self.bottleneck = BasicUformerLayer(
            dim=embed_dim*16, depth=depths[4], num_heads=num_heads[4],
            win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:4]):sum(depths[:5])], norm_layer=norm_layer, shift_flag=shift_flag
        )
        
        # Decoder
        self.upsample_0 = Upsample(embed_dim*16, embed_dim*8)
        self.decoderlayer_0 = BasicUformerLayer(
            dim=embed_dim*16, depth=depths[5], num_heads=num_heads[5],
            win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:5]):sum(depths[:6])], norm_layer=norm_layer, shift_flag=shift_flag
        )
        
        self.upsample_1 = Upsample(embed_dim*16, embed_dim*4)
        self.decoderlayer_1 = BasicUformerLayer(
            dim=embed_dim*8, depth=depths[6], num_heads=num_heads[6],
            win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:6]):sum(depths[:7])], norm_layer=norm_layer, shift_flag=shift_flag
        )
        
        self.upsample_2 = Upsample(embed_dim*8, embed_dim*2)
        self.decoderlayer_2 = BasicUformerLayer(
            dim=embed_dim*4, depth=depths[7], num_heads=num_heads[7],
            win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:7]):sum(depths[:8])], norm_layer=norm_layer, shift_flag=shift_flag
        )
        
        self.upsample_3 = Upsample(embed_dim*4, embed_dim)
        self.decoderlayer_3 = BasicUformerLayer(
            dim=embed_dim*2, depth=depths[8], num_heads=num_heads[8],
            win_size=win_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:8]):sum(depths[:9])], norm_layer=norm_layer, shift_flag=shift_flag
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x):
        """Pad image to be divisible by window size * 16"""
        _, _, h, w = x.size()
        padder_size = self.win_size * 16
        mod_pad_h = (padder_size - h % padder_size) % padder_size
        mod_pad_w = (padder_size - w % padder_size) % padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.check_image_size(x)
        _, _, H_pad, W_pad = x.shape
        
        # Input projection
        y = self.input_proj(x)
        y = self.pos_drop(y)
        
        # Encoder
        conv0 = self.encoderlayer_0(y)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0)
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1)
        pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2)
        pool3 = self.dowsample_3(conv3)
        
        # Bottleneck
        conv4 = self.bottleneck(pool3)
        
        # Decoder
        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat([up0, conv3], -1)
        deconv0 = self.decoderlayer_0(deconv0)
        
        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1)
        
        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2)
        
        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3, conv0], -1)
        deconv3 = self.decoderlayer_3(deconv3)
        
        # Output projection
        y = self.output_proj(deconv3)
        
        # Residual connection and crop to original size
        return (x + y)[:, :, :H, :W]


def Uformer_B():
    """Uformer-B (Base) configuration"""
    return Uformer(
        embed_dim=32,
        depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
        win_size=8,
        mlp_ratio=4.
    )


def Uformer_S():
    """Uformer-S (Small) configuration - lighter version"""
    return Uformer(
        embed_dim=16,
        depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        num_heads=[1, 2, 4, 8, 8, 8, 4, 2, 1],
        win_size=8,
        mlp_ratio=4.
    )


if __name__ == '__main__':
    # Test the model
    model = Uformer()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
