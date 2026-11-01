"""
Post-processing of synthetic SAR tensor outputs into visual displays, calibrated dB arrays, and export formats.
"""
import io
import numpy as np
from PIL import Image
import tifffile
from typing import Dict, Any, Tuple
from ..preprocessing.normalization import denormalize_sar, sar_to_display_image

def process_sar_prediction(raw_tensor_norm: np.ndarray) -> Dict[str, Any]:
    """
    Process normalized synthetic SAR output array in [-1.0, 1.0].
    Returns:
        Dict containing:
        - visual_image: 8-bit numpy array (H, W) or (H, W, 3)
        - calibrated_db: Float32 numpy array with dB backscatter values
        - stats: min, max, mean, std in dB
    """
    norm_arr = np.clip(raw_tensor_norm, -1.0, 1.0)
    
    # 1. Calibrated dB backscatter estimation
    calibrated_db = denormalize_sar(norm_arr)
    
    # 2. Visual 8-bit display image
    visual_img = sar_to_display_image(norm_arr)
    
    stats = {
        "min_db": round(float(np.min(calibrated_db)), 2),
        "max_db": round(float(np.max(calibrated_db)), 2),
        "mean_db": round(float(np.mean(calibrated_db)), 2),
        "std_db": round(float(np.std(calibrated_db)), 2),
    }
    
    return {
        "visual_image": visual_img,
        "calibrated_db": calibrated_db,
        "stats": stats
    }

def export_as_png_bytes(visual_array: np.ndarray) -> bytes:
    """Export 8-bit visual array as PNG byte buffer."""
    buf = io.BytesIO()
    im = Image.fromarray(visual_array)
    im.save(buf, format="PNG")
    return buf.getvalue()

def export_as_geotiff_bytes(calibrated_array: np.ndarray) -> bytes:
    """Export calibrated dB backscatter as GeoTIFF bytes."""
    buf = io.BytesIO()
    tifffile.imwrite(buf, calibrated_array.astype(np.float32))
    return buf.getvalue()
