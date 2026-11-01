"""
Robust image loading supporting multi-band GeoTIFF, PNG, JPEG with band extraction.
"""
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image
from typing import Dict, Any, Tuple, Optional
from .normalization import normalize_optical, normalize_sar
from ..utils.logging_config import logger

def load_optical_image(file_path: Path, target_size: Optional[Tuple[int, int]] = (256, 256)) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load an optical image file and extract standard 3-channel RGB.
    Returns:
        Tuple of (normalized_array_in_range_[-1,1], metadata_dict)
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Optical image not found: {p}")
        
    ext = p.suffix.lower()
    meta = {"filename": p.name, "format": ext.replace(".", "").upper(), "raw_shape": None}
    
    if ext in [".tif", ".tiff"]:
        raw = tifffile.imread(str(p))
        meta["raw_shape"] = raw.shape
        meta["raw_dtype"] = str(raw.dtype)
        
        # Check channel structure
        if raw.ndim == 2:
            # Grayscale -> replicate to 3 channels
            rgb = np.stack([raw, raw, raw], axis=-1)
        elif raw.ndim == 3:
            num_channels = raw.shape[-1] if raw.shape[-1] < raw.shape[0] else raw.shape[0]
            if raw.shape[0] == num_channels: # Channels first (C, H, W)
                raw = np.transpose(raw, (1, 2, 0))
                
            if raw.shape[-1] == 13:
                # Sentinel-2 MSI: Extract True Color Composite: B4 (Red: idx 3), B3 (Green: idx 2), B2 (Blue: idx 1)
                rgb = np.stack([raw[..., 3], raw[..., 2], raw[..., 1]], axis=-1)
            elif raw.shape[-1] >= 3:
                rgb = raw[..., :3]
            elif raw.shape[-1] == 1:
                rgb = np.repeat(raw, 3, axis=-1)
            else:
                rgb = np.repeat(raw[..., :1], 3, axis=-1)
        else:
            raise ValueError(f"Unsupported optical array shape: {raw.shape}")
    else:
        with Image.open(str(p)) as im:
            im = im.convert("RGB")
            raw = np.array(im)
            rgb = raw
            meta["raw_shape"] = rgb.shape
            meta["raw_dtype"] = str(rgb.dtype)
            
    # For high-bit depth remote sensing images (>255), apply 2% - 98% percentile stretch to [0, 255]
    if rgb.max() > 255.0:
        valid = rgb[rgb > 0]
        if len(valid) > 10:
            p2, p98 = np.percentile(valid, (2, 98))
            if p98 > p2:
                rgb = np.clip((rgb - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.float32)
            else:
                rgb = np.clip(rgb / 3000.0 * 255.0, 0, 255).astype(np.float32)
        else:
            rgb = np.clip(rgb / 3000.0 * 255.0, 0, 255).astype(np.float32)
            
    # Resize if needed
    if target_size and (rgb.shape[0] != target_size[0] or rgb.shape[1] != target_size[1]):
        pil_im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        pil_im = pil_im.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
        rgb = np.array(pil_im, dtype=np.float32)
        
    norm_arr = normalize_optical(rgb)
    return norm_arr, meta

def load_sar_image(file_path: Path, target_size: Optional[Tuple[int, int]] = (256, 256), target_channels: int = 1) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a SAR reference image and extract target channels (1-channel VV/amplitude or 2-channel VV/VH).
    Returns:
        Tuple of (normalized_array_in_range_[-1,1], metadata_dict)
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"SAR image not found: {p}")
        
    ext = p.suffix.lower()
    meta = {"filename": p.name, "format": ext.replace(".", "").upper(), "raw_shape": None}
    
    if ext in [".tif", ".tiff"]:
        raw = tifffile.imread(str(p))
        meta["raw_shape"] = raw.shape
        meta["raw_dtype"] = str(raw.dtype)
        
        if raw.ndim == 2:
            sar = raw[..., np.newaxis]
        elif raw.ndim == 3:
            if raw.shape[0] < raw.shape[-1]: # Channels first
                raw = np.transpose(raw, (1, 2, 0))
            if target_channels == 1:
                # Primary polarization VV
                sar = raw[..., :1]
            else:
                sar = raw[..., :target_channels]
        else:
            raise ValueError(f"Unsupported SAR array shape: {raw.shape}")
    else:
        with Image.open(str(p)) as im:
            raw = np.array(im)
            meta["raw_shape"] = raw.shape
            meta["raw_dtype"] = str(raw.dtype)
            if raw.ndim == 2:
                sar = raw[..., np.newaxis]
            else:
                sar = raw[..., :target_channels] if raw.ndim == 3 else raw[..., np.newaxis]
                
    # Resize if needed
    if target_size and (sar.shape[0] != target_size[0] or sar.shape[1] != target_size[1]):
        # Per-channel resize
        resized_channels = []
        for c in range(sar.shape[-1]):
            ch = sar[..., c]
            pil_im = Image.fromarray(ch)
            pil_im = pil_im.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
            resized_channels.append(np.array(pil_im))
        sar = np.stack(resized_channels, axis=-1)
        
    norm_arr = normalize_sar(sar)
    return norm_arr, meta
