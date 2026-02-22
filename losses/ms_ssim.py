"""
Multi-Scale Structural Similarity (MS-SSIM) Loss
Used in STRRNet training with weight 0.2

Uses pytorch-msssim for stable implementation.
Install: pip install pytorch-msssim

Reference:
- Wang, Z., Simoncelli, E.P., Bovik, A.C.: Multiscale structural similarity
  for image quality assessment. In: IEEE Asilomar Conference (2003)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to use pytorch-msssim (more stable)
try:
    from pytorch_msssim import ssim as pt_ssim, ms_ssim as pt_ms_ssim
    USE_PYTORCH_MSSSIM = True
except ImportError:
    USE_PYTORCH_MSSSIM = False
    print("Warning: pytorch-msssim not found. Using fallback implementation.")
    print("Install with: pip install pytorch-msssim")


def gaussian_kernel(kernel_size=11, sigma=1.5, channels=3):
    """Create 2D Gaussian kernel"""
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel


def ssim(img1, img2, kernel, kernel_size=11, reduction='mean'):
    """Calculate SSIM between two images"""
    C = img1.size(1)
    padding = kernel_size // 2
    
    mu1 = F.conv2d(img1, kernel, padding=padding, groups=C)
    mu2 = F.conv2d(img2, kernel, padding=padding, groups=C)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=padding, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=padding, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=padding, groups=C) - mu1_mu2
    
    # Clamp to avoid negative variance (numerical stability)
    sigma1_sq = torch.clamp(sigma1_sq, min=0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0)
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if reduction == 'mean':
        return ssim_map.mean()
    elif reduction == 'sum':
        return ssim_map.sum()
    else:
        return ssim_map


def ms_ssim(img1, img2, kernel, kernel_size=11, weights=None, reduction='mean'):
    """Calculate Multi-Scale SSIM (fallback implementation)"""
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    
    weights = torch.tensor(weights, device=img1.device, dtype=img1.dtype)
    levels = len(weights)
    
    C = img1.size(1)
    padding = kernel_size // 2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    msssim_vals = []
    mcs_vals = []
    
    for i in range(levels):
        # Mean
        mu1 = F.conv2d(img1, kernel, padding=padding, groups=C)
        mu2 = F.conv2d(img2, kernel, padding=padding, groups=C)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Variance & Covariance
        sigma1_sq = F.conv2d(img1 * img1, kernel, padding=padding, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, kernel, padding=padding, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, kernel, padding=padding, groups=C) - mu1_mu2
        
        # SSIM components
        luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        contrast_structure = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        
        if i < levels - 1:
            mcs_vals.append(contrast_structure.mean())
            # Downsample
            img1 = F.avg_pool2d(img1, 2)
            img2 = F.avg_pool2d(img2, 2)
        else:
            # Last level: include luminance
            msssim_vals.append((luminance * contrast_structure).mean())
    
    # Combine scales
    mcs_vals = torch.stack(mcs_vals)
    msssim_vals = torch.stack(msssim_vals)
    
    # Product of CS values raised to power of weights
    overall_mcs = torch.prod(mcs_vals.pow(weights[:-1]))
    overall_ssim = msssim_vals[-1].pow(weights[-1])
    
    ms_ssim_val = overall_mcs * overall_ssim
    
    return ms_ssim_val


class SSIMLoss(nn.Module):
    """
    SSIM Loss: 1 - SSIM
    """
    def __init__(self, kernel_size=11, sigma=1.5, channels=3):
        super(SSIMLoss, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        
        # Pre-compute kernel
        kernel = gaussian_kernel(kernel_size, sigma, channels)
        self.register_buffer('kernel', kernel)
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] predicted image
            target: [B, C, H, W] target image
        
        Returns:
            1 - SSIM (lower is better)
        """
        ssim_val = ssim(pred, target, self.kernel, self.kernel_size)
        return 1.0 - ssim_val


class MSSSIMLoss(nn.Module):
    """
    Multi-Scale SSIM Loss: 1 - MS-SSIM
    
    Used in STRRNet with weight 0.2
    Uses pytorch-msssim if available (more stable).
    """
    def __init__(self, kernel_size=11, sigma=1.5, channels=3, weights=None):
        super(MSSSIMLoss, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        self.use_pytorch_msssim = USE_PYTORCH_MSSSIM
        
        if weights is None:
            self.weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        else:
            self.weights = weights
        
        # Pre-compute kernel for fallback
        if not self.use_pytorch_msssim:
            kernel = gaussian_kernel(kernel_size, sigma, channels)
            self.register_buffer('kernel', kernel)
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] predicted image (range [0, 1])
            target: [B, C, H, W] target image (range [0, 1])
        
        Returns:
            1 - MS-SSIM (lower is better)
        """
        # Ensure input is in [0, 1] range and float32 for stability
        pred = pred.float().clamp(0, 1)
        target = target.float().clamp(0, 1)
        
        if self.use_pytorch_msssim:
            # Use pytorch-msssim (stable implementation)
            # win_size must be odd and >= 3, default 11
            # For 128x128 images, use fewer levels to avoid size issues
            ms_ssim_val = pt_ms_ssim(pred, target, data_range=1.0, size_average=True, win_size=7)
        else:
            # Fallback implementation
            ms_ssim_val = ms_ssim(pred, target, self.kernel, self.kernel_size, self.weights)
        
        # Clamp to avoid NaN from negative values
        loss = 1.0 - ms_ssim_val.clamp(0, 1)
        
        # Check for NaN and return 0 if detected
        if torch.isnan(loss):
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        return loss


class MixedLoss(nn.Module):
    """
    Combined L1 + MS-SSIM Loss (as used in STRRNet)
    
    Loss = l1_weight * L1 + msssim_weight * (1 - MS-SSIM)
    
    Default: l1_weight=1.0, msssim_weight=0.2
    """
    def __init__(self, l1_weight=1.0, msssim_weight=0.2, kernel_size=11, sigma=1.5, channels=3):
        super(MixedLoss, self).__init__()
        self.l1_weight = l1_weight
        self.msssim_weight = msssim_weight
        
        self.l1_loss = nn.L1Loss()
        self.msssim_loss = MSSSIMLoss(kernel_size, sigma, channels)
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] predicted image
            target: [B, C, H, W] target image
        
        Returns:
            Combined loss value
        """
        l1 = self.l1_loss(pred, target)
        msssim = self.msssim_loss(pred, target)
        
        total_loss = self.l1_weight * l1 + self.msssim_weight * msssim
        
        return total_loss, l1, msssim


if __name__ == '__main__':
    # Test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test images
    img1 = torch.rand(2, 3, 128, 128, device=device)
    img2 = torch.rand(2, 3, 128, 128, device=device)
    
    # Test SSIM Loss
    ssim_loss = SSIMLoss().to(device)
    loss_ssim = ssim_loss(img1, img2)
    print(f"SSIM Loss: {loss_ssim.item():.4f}")
    
    # Test MS-SSIM Loss
    msssim_loss = MSSSIMLoss().to(device)
    loss_msssim = msssim_loss(img1, img2)
    print(f"MS-SSIM Loss: {loss_msssim.item():.4f}")
    
    # Test Mixed Loss
    mixed_loss = MixedLoss().to(device)
    total, l1, ms = mixed_loss(img1, img2)
    print(f"Mixed Loss: {total.item():.4f} (L1: {l1.item():.4f}, MS-SSIM: {ms.item():.4f})")
    
    # Test with similar images
    img2_similar = img1 + 0.01 * torch.randn_like(img1)
    loss_similar = msssim_loss(img1, img2_similar)
    print(f"MS-SSIM Loss (similar images): {loss_similar.item():.4f}")
