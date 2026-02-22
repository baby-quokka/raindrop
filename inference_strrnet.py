"""
STRRNet Inference Script
NTIRE 2025 Challenge on Day and Night Raindrop Removal - 1st Place (Miracle Team)

Testing Strategy (from paper):
1. Sliding window inference: 128x128 with overlap 32
2. Median fusion across frames in same scene
3. Weighted sum of median image and original output

Usage:
    python inference_strrnet.py --checkpoint path/to/model.pkl --input_dir path/to/images
"""

import os
import argparse
import glob
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm

from models.STRRNet import STRRNet


def load_image(path):
    """Load image and convert to tensor [1, 3, H, W]."""
    img = Image.open(path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img


def save_image(tensor, path):
    """Save tensor [1, 3, H, W] as image."""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def sliding_window_inference(model, image, window_size=128, overlap=32, device='cuda'):
    """
    Apply model using sliding window approach.
    
    Args:
        model: STRRNet model
        image: [1, 3, H, W] input image
        window_size: sliding window size (default 128)
        overlap: overlap between windows (default 32)
        device: computation device
    
    Returns:
        [1, 3, H, W] output image
    """
    _, _, H, W = image.shape
    stride = window_size - overlap
    
    # Pad image if necessary
    pad_h = (stride - (H - window_size) % stride) % stride
    pad_w = (stride - (W - window_size) % stride) % stride
    
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
    
    _, _, H_pad, W_pad = image.shape
    
    # Output accumulator and weight map
    output = torch.zeros_like(image)
    weight = torch.zeros((1, 1, H_pad, W_pad), device=device)
    
    # Create weight kernel for blending (higher weight in center)
    kernel = torch.ones((1, 1, window_size, window_size), device=device)
    # Gaussian-like weighting: higher in center
    x = torch.linspace(-1, 1, window_size, device=device)
    y = torch.linspace(-1, 1, window_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    gaussian_weight = torch.exp(-(xx**2 + yy**2) / 0.5)
    kernel = gaussian_weight.unsqueeze(0).unsqueeze(0)
    
    # Sliding window
    for i in range(0, H_pad - window_size + 1, stride):
        for j in range(0, W_pad - window_size + 1, stride):
            # Extract window
            window = image[:, :, i:i+window_size, j:j+window_size].to(device)
            
            # Process window
            with torch.no_grad():
                pred = model(window, semantic_labels=None)
            
            # Accumulate
            output[:, :, i:i+window_size, j:j+window_size] += pred * kernel
            weight[:, :, i:i+window_size, j:j+window_size] += kernel
    
    # Normalize
    output = output / (weight + 1e-8)
    
    # Remove padding
    output = output[:, :, :H, :W]
    
    return output


def median_fusion(images):
    """
    Apply median fusion across multiple images.
    
    Since raindrop positions vary across frames while background is consistent,
    median fusion removes inconsistent artifacts.
    
    Args:
        images: list of [1, 3, H, W] tensors
    
    Returns:
        [1, 3, H, W] median-fused image
    """
    if len(images) == 1:
        return images[0]
    
    stacked = torch.cat(images, dim=0)  # [N, 3, H, W]
    median, _ = torch.median(stacked, dim=0, keepdim=True)
    return median


def weighted_fusion(original, median, alpha=0.7):
    """
    Weighted sum of original output and median-fused image.
    
    Args:
        original: [1, 3, H, W] model output
        median: [1, 3, H, W] median-fused image
        alpha: weight for original (default 0.7)
    
    Returns:
        [1, 3, H, W] final output
    """
    return alpha * original + (1 - alpha) * median


def extract_scene_id(path):
    """Extract scene ID from file path."""
    # e.g., /Drop/00124/00005.png -> 00124
    parts = path.replace('\\', '/').split('/')
    for i, part in enumerate(parts):
        if part == 'Drop' and i + 1 < len(parts):
            return parts[i + 1]
    return os.path.dirname(path)


def inference_single(model, image_path, output_path, device, use_sliding_window=True):
    """Process single image."""
    image = load_image(image_path).to(device)
    
    with torch.no_grad():
        if use_sliding_window:
            output = sliding_window_inference(model, image, device=device)
        else:
            # Standard inference with padding
            _, _, h, w = image.shape
            factor = 32
            H = ((h + factor) // factor) * factor
            W = ((w + factor) // factor) * factor
            padh, padw = H - h, W - w
            
            image_padded = F.pad(image, (0, padw, 0, padh), 'reflect')
            output = model(image_padded, semantic_labels=None)
            output = output[:, :, :h, :w]
    
    # Clamp to valid range
    output = torch.clamp(output, 0, 1)
    
    save_image(output, output_path)
    return output


def inference_with_median_fusion(model, image_paths, output_dir, device):
    """
    Process images with median fusion for same scene.
    
    Args:
        model: STRRNet model
        image_paths: list of image paths
        output_dir: output directory
        device: computation device
    """
    # Group images by scene
    scene_images = defaultdict(list)
    for path in image_paths:
        scene_id = extract_scene_id(path)
        scene_images[scene_id].append(path)
    
    # Process each scene
    for scene_id, paths in tqdm(scene_images.items(), desc="Processing scenes"):
        outputs = []
        
        # First pass: get individual outputs
        for path in paths:
            image = load_image(path).to(device)
            with torch.no_grad():
                output = sliding_window_inference(model, image, device=device)
            outputs.append(output)
        
        # Median fusion
        if len(outputs) > 1:
            median = median_fusion(outputs)
        else:
            median = outputs[0]
        
        # Save results with weighted fusion
        for i, (path, output) in enumerate(zip(paths, outputs)):
            # Weighted fusion of individual output and median
            if len(outputs) > 1:
                final = weighted_fusion(output, median, alpha=0.7)
            else:
                final = output
            
            final = torch.clamp(final, 0, 1)
            
            # Save
            filename = os.path.basename(path)
            save_path = os.path.join(output_dir, filename)
            save_image(final, save_path)


def main():
    parser = argparse.ArgumentParser(description='STRRNet Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input image directory')
    parser.add_argument('--output_dir', type=str, default='results/strrnet_output',
                        help='Output directory')
    parser.add_argument('--use_sliding_window', action='store_true', default=True,
                        help='Use sliding window inference')
    parser.add_argument('--use_median_fusion', action='store_true', default=False,
                        help='Use median fusion across scene frames')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = STRRNet(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        use_semantic_guidance=True,
        use_bg_subnet=True
    )
    
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {params:.2f}M")
    
    # Get input images
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, '**', ext), recursive=True))
    
    print(f"Found {len(image_paths)} images")
    
    if args.use_median_fusion:
        # Process with median fusion
        inference_with_median_fusion(model, image_paths, args.output_dir, device)
    else:
        # Process individually
        for path in tqdm(image_paths, desc="Processing"):
            filename = os.path.basename(path)
            output_path = os.path.join(args.output_dir, filename)
            inference_single(model, path, output_path, device, args.use_sliding_window)
    
    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
