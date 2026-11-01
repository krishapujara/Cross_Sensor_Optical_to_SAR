"""
Scientific validation routines for optical-SAR image pairs and single inputs.
"""
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image
from typing import Dict, Any, Tuple
from ..utils.logging_config import logger

def validate_image_file(file_path: Path) -> Dict[str, Any]:
    """Validate readability, channels, shape, and value ranges of an image file."""
    p = Path(file_path)
    if not p.exists():
        return {"valid": False, "error": f"File does not exist: {p.name}"}
    if p.stat().st_size == 0:
        return {"valid": False, "error": f"File is empty: {p.name}"}
        
    ext = p.suffix.lower()
    try:
        if ext in [".tif", ".tiff"]:
            arr = tifffile.imread(str(p))
        else:
            with Image.open(str(p)) as im:
                arr = np.array(im)
                
        if arr is None or arr.size == 0:
            return {"valid": False, "error": f"Unable to parse image data from {p.name}"}
            
        # Check for NaN / Inf
        has_nan = bool(np.isnan(arr).any()) if np.issubdtype(arr.dtype, np.floating) else False
        has_inf = bool(np.isinf(arr).any()) if np.issubdtype(arr.dtype, np.floating) else False
        
        shape = list(arr.shape)
        channels = 1 if len(shape) == 2 else shape[-1] if len(shape) == 3 else shape[0]
        
        return {
            "valid": not (has_nan or has_inf),
            "file_name": p.name,
            "format": ext.replace(".", "").upper(),
            "shape": shape,
            "channels": channels,
            "dtype": str(arr.dtype),
            "min_val": float(np.min(arr)),
            "max_val": float(np.max(arr)),
            "mean_val": float(np.mean(arr)),
            "has_nan": has_nan,
            "has_inf": has_inf,
            "file_size_kb": round(p.stat().st_size / 1024, 2)
        }
    except Exception as e:
        logger.warning(f"Validation failed for {p.name}: {e}")
        return {"valid": False, "error": f"Corrupted or invalid image: {str(e)}"}

def validate_pair_correspondence(optical_path: Path, sar_path: Path) -> Dict[str, Any]:
    """Validate that optical and SAR image pair are compatible in dimensions and valid."""
    opt_val = validate_image_file(optical_path)
    sar_val = validate_image_file(sar_path)
    
    if not opt_val.get("valid"):
        return {"valid": False, "reason": f"Optical validation failed: {opt_val.get('error')}", "optical": opt_val, "sar": sar_val}
    if not sar_val.get("valid"):
        return {"valid": False, "reason": f"SAR validation failed: {sar_val.get('error')}", "optical": opt_val, "sar": sar_val}
        
    # Spatial dimension comparison (height, width)
    opt_h, opt_w = opt_val["shape"][:2]
    sar_h, sar_w = sar_val["shape"][:2]
    
    dim_match = (opt_h == sar_h) and (opt_w == sar_w)
    
    return {
        "valid": dim_match,
        "reason": "Pair dimensions match and data is clean" if dim_match else f"Dimension mismatch: Optical {opt_h}x{opt_w} vs SAR {sar_h}x{sar_w}",
        "dimension_match": dim_match,
        "optical": opt_val,
        "sar": sar_val
    }
