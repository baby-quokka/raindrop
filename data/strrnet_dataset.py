"""
STRRNet Dataset with Semantic Labels

4 Semantic Classes:
- 0: night_bg_focus (Night, Background-focused)
- 1: night_raindrop_focus (Night, Raindrop-focused)
- 2: day_bg_focus (Day, Background-focused)
- 3: day_raindrop_focus (Day, Raindrop-focused)

RaindropClarity dataset structure:
- NightRainDrop_Train: Night images
- DayRainDrop_Train: Day images
- Focus info needs to be determined from dataset metadata or folder structure
"""

import os
import random
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from data.utils import read_rgb, crop_pair, flip_pair, to_m01, to_m11


def get_semantic_label(file_path: str, focus_info: Optional[dict] = None) -> int:
    """
    Determine semantic label from file path.
    
    Labels:
    - 0: night_bg_focus
    - 1: night_raindrop_focus
    - 2: day_bg_focus
    - 3: day_raindrop_focus
    
    Args:
        file_path: path to the image file
        focus_info: optional dict mapping scene_id to focus type ('bg' or 'raindrop')
    
    Returns:
        semantic label (0-3)
    """
    path_lower = file_path.lower()
    
    # Determine day/night
    is_night = 'night' in path_lower
    
    # Determine focus type
    # RaindropClarity: scene IDs < some threshold might be bg-focused
    # This is a heuristic - adjust based on actual dataset structure
    
    # Extract scene ID from path (e.g., /Drop/00124/00005.png -> 00124)
    parts = file_path.replace('\\', '/').split('/')
    scene_id = None
    for i, part in enumerate(parts):
        if part == 'Drop' and i + 1 < len(parts):
            scene_id = parts[i + 1]
            break
    
    # Default heuristic for focus detection
    # In RaindropClarity: 
    # - Day: 1575 bg-focused, 3138 raindrop-focused (roughly scene_id threshold)
    # - Night: 4143 bg-focused, 4512 raindrop-focused
    # This needs to be calibrated with actual dataset info
    
    is_raindrop_focus = True  # Default
    
    if focus_info is not None and scene_id in focus_info:
        is_raindrop_focus = focus_info[scene_id] == 'raindrop'
    else:
        # Heuristic: use scene_id to guess focus type
        # Adjust these thresholds based on your dataset
        try:
            scene_num = int(scene_id) if scene_id else 0
            if is_night:
                # Night: approximately 4143 bg-focused scenes
                is_raindrop_focus = scene_num >= 100  # Adjust threshold
            else:
                # Day: approximately 1575 bg-focused scenes  
                is_raindrop_focus = scene_num >= 50  # Adjust threshold
        except:
            is_raindrop_focus = True
    
    # Compute label
    if is_night:
        return 1 if is_raindrop_focus else 0  # night_raindrop or night_bg
    else:
        return 3 if is_raindrop_focus else 2  # day_raindrop or day_bg


class STRRNetDataset(Dataset):
    """
    Dataset for STRRNet training with semantic labels.
    
    Args:
        filename: path to file list (.txt)
        crop: crop size for training
        is_test: whether this is test mode
        value_range: (0, 1) or (-1, 1)
        focus_info_file: optional file with focus information per scene
    """
    def __init__(self,
                 filename: str = "data/train/RaindropClarity.txt",
                 crop: int = 128,  # STRRNet uses 128x128
                 is_test: bool = False,
                 value_range: Tuple = (0, 1),
                 focus_info_file: Optional[str] = None):
        super().__init__()
        self.filename = filename
        self.crop = crop
        self.is_test = is_test
        self.value_range = value_range
        self.data = os.path.splitext(os.path.basename(self.filename))[0]
        
        # Load focus info if provided
        self.focus_info = None
        if focus_info_file and os.path.exists(focus_info_file):
            self.focus_info = self._load_focus_info(focus_info_file)
        
        # Read file list
        with open(self.filename, "r") as f:
            self.input_files: List[str] = [ln.strip() for ln in f if ln.strip()]
        
        # Build file pairs with semantic labels
        self.files = []
        for input_path in self.input_files:
            # Get GT path
            if "RaindropClarity" in self.data or "/Drop/" in input_path:
                gt_path = input_path.replace('/Drop/', '/Clear/')
            else:
                gt_path = input_path
            
            # Get semantic label
            semantic_label = get_semantic_label(input_path, self.focus_info)
            
            self.files.append((input_path, gt_path, semantic_label))
    
    def _load_focus_info(self, focus_info_file: str) -> dict:
        """Load focus information from file."""
        focus_info = {}
        with open(focus_info_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    scene_id, focus_type = parts[0], parts[1]
                    focus_info[scene_id] = focus_type
        return focus_info
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        input_path, gt_path, semantic_label = self.files[idx]
        
        # Read images
        a = read_rgb(input_path)  # uint8 [3, H, W]
        b = read_rgb(gt_path)     # uint8 [3, H, W]
        
        if not self.is_test:
            # Training: crop and augment
            a, b = crop_pair(a, b, self.crop)
            a, b = flip_pair(a, b, p=0.5)
            
            # Random rotation (geometric augmentation as in STRRNet)
            if random.random() < 0.5:
                k = random.choice([1, 2, 3])  # 90, 180, 270 degrees
                a = torch.rot90(a, k, [1, 2])
                b = torch.rot90(b, k, [1, 2])
            
            # Convert to desired range
            if self.value_range == (0, 1):
                return to_m01(a), to_m01(b), semantic_label
            elif self.value_range == (-1, 1):
                return to_m11(a), to_m11(b), semantic_label
            else:
                raise ValueError(f"Unsupported value_range {self.value_range}")
        else:
            # Test: return full image with name
            name = os.path.splitext(os.path.basename(input_path))[0]
            if self.value_range == (0, 1):
                return to_m01(a), to_m01(b), semantic_label, name
            elif self.value_range == (-1, 1):
                return to_m11(a), to_m11(b), semantic_label, name
            else:
                raise ValueError(f"Unsupported value_range {self.value_range}")


def strrnet_train_dataloader(
    filename: str = "data/train/RaindropClarity.txt",
    value_range: Tuple = (0, 1),
    crop: int = 128,  # STRRNet default
    batch_size: int = 8,
    num_workers: int = 4,
    focus_info_file: Optional[str] = None
):
    """Create training dataloader for STRRNet."""
    dataloader = DataLoader(
        STRRNetDataset(
            filename,
            crop=crop,
            is_test=False,
            value_range=value_range,
            focus_info_file=focus_info_file
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    return dataloader


def strrnet_test_dataloader(
    filename: str = "data/test/RaindropClarity.txt",
    value_range: Tuple = (0, 1),
    batch_size: int = 1,
    num_workers: int = 1,
    focus_info_file: Optional[str] = None
):
    """Create test dataloader for STRRNet."""
    dataloader = DataLoader(
        STRRNetDataset(
            filename,
            is_test=True,
            value_range=value_range,
            focus_info_file=focus_info_file
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )
    return dataloader


if __name__ == '__main__':
    # Test
    import os
    
    # Check if data file exists
    data_file = "data/train/RaindropClarity.txt"
    if os.path.exists(data_file):
        dataset = STRRNetDataset(data_file, crop=128)
        print(f"Dataset size: {len(dataset)}")
        
        # Get a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Input shape: {sample[0].shape}")
            print(f"Target shape: {sample[1].shape}")
            print(f"Semantic label: {sample[2]}")
            
            # Count labels
            label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            for _, _, label in dataset.files:
                label_counts[label] += 1
            
            print(f"\nLabel distribution:")
            print(f"  night_bg_focus (0): {label_counts[0]}")
            print(f"  night_raindrop_focus (1): {label_counts[1]}")
            print(f"  day_bg_focus (2): {label_counts[2]}")
            print(f"  day_raindrop_focus (3): {label_counts[3]}")
    else:
        print(f"Data file not found: {data_file}")
