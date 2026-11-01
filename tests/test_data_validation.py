"""
Unit tests for data validation module.
"""
import pytest
import numpy as np
import tifffile
from PIL import Image
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.data_validation import validate_image_file, validate_pair_correspondence

def test_validate_missing_file(tmp_path):
    """Missing file should return valid=False with helpful error message."""
    missing = tmp_path / "non_existent.tif"
    res = validate_image_file(missing)
    assert res["valid"] is False
    assert "does not exist" in res["error"]

def test_validate_valid_tiff(tmp_path):
    """Valid GeoTIFF should return valid=True with correct dimensions."""
    fpath = tmp_path / "test_valid.tif"
    arr = np.random.uniform(-30, 0, (64, 64, 2)).astype(np.float32)
    tifffile.imwrite(str(fpath), arr)
    
    res = validate_image_file(fpath)
    assert res["valid"] is True
    assert res["shape"] == [64, 64, 2]
    assert res["has_nan"] is False

def test_pair_correspondence_mismatch(tmp_path):
    """Dimension mismatch between optical and SAR should be flagged as invalid."""
    opt_path = tmp_path / "opt.png"
    sar_path = tmp_path / "sar.png"
    
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(opt_path)
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(sar_path)
    
    val = validate_pair_correspondence(opt_path, sar_path)
    assert val["valid"] is False
    assert val["dimension_match"] is False
