"""
Geospatial metadata extraction and preservation (CRS, transform, bounds, resolution).
"""
from pathlib import Path
from typing import Dict, Any, Optional
import tifffile
from ..utils.logging_config import logger

def extract_geospatial_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Extract geospatial metadata from GeoTIFF if present.
    Gracefully handles standard non-georeferenced images.
    """
    p = Path(file_path)
    meta = {
        "is_geospatial": False,
        "crs": None,
        "transform": None,
        "bounds": None,
        "resolution": None,
        "nodata": None
    }
    
    if p.suffix.lower() not in [".tif", ".tiff"]:
        return meta
        
    try:
        # Check using rasterio if available
        try:
            import rasterio
            with rasterio.open(str(p)) as src:
                meta["is_geospatial"] = bool(src.crs is not None)
                meta["crs"] = str(src.crs) if src.crs else "Unprojected / Local"
                meta["transform"] = [float(x) for x in src.transform] if src.transform else None
                meta["bounds"] = {
                    "left": float(src.bounds.left),
                    "bottom": float(src.bounds.bottom),
                    "right": float(src.bounds.right),
                    "top": float(src.bounds.top)
                } if src.bounds else None
                meta["resolution"] = [float(src.res[0]), float(src.res[1])] if src.res else None
                meta["nodata"] = float(src.nodata) if src.nodata is not None else None
                return meta
        except ImportError:
            # Fallback to tifffile tags
            with tifffile.TiffFile(str(p)) as tf:
                page = tf.pages[0]
                geotags = page.geotiff_tags
                if geotags:
                    meta["is_geospatial"] = True
                    meta["crs"] = f"GeoKeyDirectory: {len(geotags)} tags found"
                    meta["tags"] = {k: str(v) for k, v in list(geotags.items())[:5]}
                return meta
    except Exception as e:
        logger.debug(f"Geospatial tag read non-critical error: {e}")
        return meta
