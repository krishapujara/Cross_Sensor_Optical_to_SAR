"""
Domain-specific image quality metrics for optical-to-SAR evaluation (PSNR, SSIM, MAE, MSE).
"""
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim
from typing import Dict, Any, Union

def calculate_mse(reference: np.ndarray, generated: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((reference.astype(np.float64) - generated.astype(np.float64)) ** 2))

def calculate_mae(reference: np.ndarray, generated: np.ndarray) -> float:
    """Mean Absolute Error (L1 error)."""
    return float(np.mean(np.abs(reference.astype(np.float64) - generated.astype(np.float64))))

def calculate_psnr(reference: np.ndarray, generated: np.ndarray, data_range: float = 2.0) -> float:
    """
    Peak Signal-to-Noise Ratio (in dB).
    data_range: 2.0 if arrays are in [-1, 1], or 1.0 if [0, 1], or 255 if [0, 255].
    """
    mse = calculate_mse(reference, generated)
    if mse == 0.0:
        return 100.0
    return float(10.0 * np.log10((data_range ** 2) / mse))

def calculate_ssim(reference: np.ndarray, generated: np.ndarray, data_range: float = 2.0) -> float:
    """
    Structural Similarity Index (SSIM).
    Measures luminance, contrast, and structural correlation.
    """
    ref = reference.squeeze()
    gen = generated.squeeze()
    
    # If 3D (H, W, C), specify channel_axis
    if ref.ndim == 3:
        ch_axis = -1 if ref.shape[-1] <= 4 else 0
        return float(skimage_ssim(ref, gen, data_range=data_range, channel_axis=ch_axis))
    elif ref.ndim == 2:
        return float(skimage_ssim(ref, gen, data_range=data_range))
    else:
        # Fallback
        return float(skimage_ssim(ref.reshape(-1, ref.shape[-1]), gen.reshape(-1, gen.shape[-1]), data_range=data_range))

def evaluate_image_pair(
    reference: Union[np.ndarray, torch.Tensor],
    generated: Union[np.ndarray, torch.Tensor],
    data_range: float = 2.0
) -> Dict[str, float]:
    """
    Compute full evaluation metric suite for a single (Reference SAR, Synthetic SAR) pair.
    Inputs can be tensors or arrays in range [-1.0, 1.0].
    """
    if isinstance(reference, torch.Tensor):
        ref_np = reference.detach().cpu().numpy()
    else:
        ref_np = np.asarray(reference)
        
    if isinstance(generated, torch.Tensor):
        gen_np = generated.detach().cpu().numpy()
    else:
        gen_np = np.asarray(generated)
        
    mse = calculate_mse(ref_np, gen_np)
    mae = calculate_mae(ref_np, gen_np)
    psnr = calculate_psnr(ref_np, gen_np, data_range=data_range)
    try:
        ssim = calculate_ssim(ref_np, gen_np, data_range=data_range)
    except Exception:
        # In case spatial size is too small for default window (7x7)
        ssim = 0.0
        
    return {
        "psnr": round(psnr, 3),
        "ssim": round(ssim, 4),
        "mae": round(mae, 4),
        "mse": round(mse, 4)
    }

def compute_difference_map(reference: np.ndarray, generated: np.ndarray) -> np.ndarray:
    """
    Compute pixel-wise absolute residual error map |Reference - Generated|.
    Returns 2D float array in range [0, 1].
    """
    ref = reference.squeeze()
    gen = generated.squeeze()
    diff = np.abs(ref.astype(np.float32) - gen.astype(np.float32))
    if diff.ndim == 3:
        diff = np.mean(diff, axis=-1)
    # Normalize diff to [0, 1] for visualization
    max_d = np.max(diff)
    if max_d > 0:
        diff = diff / max_d
    return diff
