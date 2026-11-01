"""
Scientific normalization and denormalization routines for optical and SAR satellite imagery.
"""
import numpy as np
import torch
from ..config.settings import SAR_MIN_DB, SAR_MAX_DB

def normalize_optical(arr: np.ndarray) -> np.ndarray:
    """
    Normalize optical image to [-1.0, 1.0].
    Handles both standard 8-bit [0, 255] and 12/16-bit Sentinel-2 surface reflectance [0, 10000].
    Uses 2% to 98% percentile contrast stretching on multispectral surface reflectance to
    maximize visual feature discriminability for deep neural networks.
    """
    arr = arr.astype(np.float32)
    max_val = np.max(arr)
    
    if max_val > 255.0:
        # Multispectral 12/16-bit surface reflectance (e.g. Sentinel-2 L1C/L2A)
        # Apply standard remote sensing 2% to 98% percentile linear contrast stretching
        valid_pixels = arr[arr > 0]
        if len(valid_pixels) > 10:
            p2, p98 = np.percentile(valid_pixels, (2, 98))
            if p98 > p2:
                arr = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
            else:
                arr = np.clip(arr / 3000.0, 0.0, 1.0)
        else:
            arr = np.clip(arr / 3000.0, 0.0, 1.0)
    else:
        # Standard 8-bit RGB [0, 255]
        arr = np.clip(arr / 255.0, 0.0, 1.0)
        
    # Map [0, 1] to [-1, 1]
    return (arr * 2.0) - 1.0

def denormalize_optical(arr: np.ndarray) -> np.ndarray:
    """Denormalize optical image from [-1.0, 1.0] to [0, 255] uint8."""
    arr = np.clip(arr, -1.0, 1.0)
    arr_01 = (arr + 1.0) / 2.0
    return (arr_01 * 255.0).astype(np.uint8)

def normalize_sar(arr: np.ndarray, min_db: float = SAR_MIN_DB, max_db: float = SAR_MAX_DB) -> np.ndarray:
    """
    Normalize SAR backscatter (in dB or amplitude) to [-1.0, 1.0].
    Default assumes Sentinel-1 sigma-0 backscatter in dB clipped between -35 dB and 0 dB.
    """
    arr = arr.astype(np.float32)
    
    # If SAR is already 8-bit [0, 255] (e.g. pre-rendered amplitude)
    if np.min(arr) >= 0.0 and np.max(arr) > 1.0:
        arr = np.clip(arr / 255.0, 0.0, 1.0)
        return (arr * 2.0) - 1.0
        
    # Standard calibrated radar backscatter in dB
    clipped = np.clip(arr, min_db, max_db)
    # Map [min_db, max_db] to [0, 1]
    norm_01 = (clipped - min_db) / (max_db - min_db)
    # Map to [-1, 1]
    return (norm_01 * 2.0) - 1.0

def denormalize_sar(arr: np.ndarray, min_db: float = SAR_MIN_DB, max_db: float = SAR_MAX_DB) -> np.ndarray:
    """Denormalize synthetic SAR from [-1.0, 1.0] back to calibrated dB backscatter."""
    arr = np.clip(arr, -1.0, 1.0)
    norm_01 = (arr + 1.0) / 2.0
    return (norm_01 * (max_db - min_db)) + min_db

def sar_to_display_image(sar_norm: np.ndarray) -> np.ndarray:
    """
    Convert normalized SAR array in [-1.0, 1.0] to a high-contrast 8-bit visual image.
    Applies adaptive histogram/contrast enhancement so radar backscatter, texture,
    and speckle characteristics are crisp and clearly visible.
    """
    arr = np.clip(sar_norm, -1.0, 1.0).astype(np.float32)
    
    # Adaptive dynamic range expansion if predictions have low variance
    val_min = float(np.min(arr))
    val_max = float(np.max(arr))
    
    if (val_max - val_min) > 0.05:
        # Real backscatter variation present: percentile stretch
        p2, p98 = np.percentile(arr, (2, 98))
        if p98 > p2:
            arr_01 = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
        else:
            arr_01 = (arr - val_min) / (val_max - val_min)
    else:
        # Fallback to direct linear [-1, 1] -> [0, 1]
        arr_01 = (arr + 1.0) / 2.0
    
    if arr_01.ndim == 2:
        return (arr_01 * 255.0).astype(np.uint8)
    elif arr_01.ndim == 3:
        if arr_01.shape[-1] == 1:
            return (arr_01[..., 0] * 255.0).astype(np.uint8)
        elif arr_01.shape[-1] == 2:
            # Sentinel-1 dual pol: R=VV, G=VH, B=VV/VH ratio
            vv = arr_01[..., 0]
            vh = arr_01[..., 1]
            ratio = np.clip(vv / (vh + 1e-4), 0.0, 1.0)
            composite = np.stack([vv, vh, ratio], axis=-1)
            return (composite * 255.0).astype(np.uint8)
        elif arr_01.shape[-1] >= 3:
            return (arr_01[..., :3] * 255.0).astype(np.uint8)
            
    return (arr_01 * 255.0).astype(np.uint8)

