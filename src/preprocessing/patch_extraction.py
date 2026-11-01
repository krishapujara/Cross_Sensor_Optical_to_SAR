"""
PyTorch Dataset, geographic split partitioning, and DataLoader construction.
"""
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Optional
from .image_loader import load_optical_image, load_sar_image
from ..utils.logging_config import logger

class PairedSatelliteDataset(Dataset):
    """PyTorch Dataset yielding paired (Optical, SAR) tensor samples."""
    
    def __init__(self, pairs: List[Dict[str, Any]], img_size: int = 256, sar_channels: int = 1, is_train: bool = True):
        self.pairs = pairs
        self.img_size = (img_size, img_size)
        self.sar_channels = sar_channels
        self.is_train = is_train
        
    def __len__(self) -> int:
        return len(self.pairs)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        item = self.pairs[idx]
        opt_path = Path(item["optical_path"])
        sar_path = Path(item["sar_path"])
        
        opt_arr, opt_meta = load_optical_image(opt_path, target_size=self.img_size)
        sar_arr, sar_meta = load_sar_image(sar_path, target_size=self.img_size, target_channels=self.sar_channels)
        
        # Data augmentation for training: random horizontal flip
        if self.is_train and random.random() > 0.5:
            opt_arr = np.fliplr(opt_arr).copy()
            sar_arr = np.fliplr(sar_arr).copy()
            
        # Convert (H, W, C) -> (C, H, W) PyTorch tensors
        opt_tensor = torch.from_numpy(opt_arr.transpose(2, 0, 1)).float()
        sar_tensor = torch.from_numpy(sar_arr.transpose(2, 0, 1)).float()
        
        meta = {
            "sample_id": item.get("sample_id", f"sample_{idx}"),
            "opt_name": opt_path.name,
            "sar_name": sar_path.name
        }
        return opt_tensor, sar_tensor, meta

def split_dataset(
    pairs: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    group_by_region: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split pairs into train, val, and test partitions with geographic leakage prevention.
    If group_by_region is True and regions are specified in metadata,
    partitions will avoid splitting identical geographic scenes across train and test.
    """
    if not pairs:
        return {"train": [], "val": [], "test": []}
        
    # Check if regions are specified
    regions = set(p.get("region") for p in pairs if p.get("region"))
    
    if group_by_region and len(regions) >= 3:
        # Geographic block split
        region_list = sorted(list(regions))
        train_region = region_list[0] # e.g. azerbaijan
        val_region = region_list[1]   # e.g. gabon
        test_region = region_list[2]  # e.g. germany
        
        train_pairs = [p for p in pairs if p.get("region") == train_region]
        val_pairs = [p for p in pairs if p.get("region") == val_region]
        test_pairs = [p for p in pairs if p.get("region") == test_region]
        
        # If any split is empty, fall back to stratified shuffle
        if not train_pairs or not val_pairs or not test_pairs:
            group_by_region = False
            
    if not group_by_region or len(regions) < 3:
        shuffled = list(pairs)
        random.seed(42)
        random.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio)) if n_total > 2 else 0
        
        train_pairs = shuffled[:n_train]
        val_pairs = shuffled[n_train:n_train + n_val]
        test_pairs = shuffled[n_train + n_val:]
        if not test_pairs and len(shuffled) > 1:
            test_pairs = val_pairs
            
    return {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs
    }

def create_dataloaders(
    split_dict: Dict[str, List[Dict[str, Any]]],
    batch_size: int = 4,
    img_size: int = 256,
    sar_channels: int = 1
) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders for each split partition."""
    loaders = {}
    for split, pairs in split_dict.items():
        if not pairs:
            continue
        ds = PairedSatelliteDataset(
            pairs,
            img_size=img_size,
            sar_channels=sar_channels,
            is_train=(split == "train")
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=min(batch_size, len(ds)),
            shuffle=(split == "train"),
            num_workers=0, # Safe for Windows
            drop_last=(split == "train" and len(ds) > batch_size)
        )
    return loaders
