"""
Unit tests for data preprocessing and normalization routines.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.normalization import (
    normalize_optical,
    denormalize_optical,
    normalize_sar,
    denormalize_sar,
    sar_to_display_image
)
from src.preprocessing.patch_extraction import split_dataset

def test_optical_normalization_roundtrip():
    """8-bit image normalized to [-1, 1] and denormalized should match within rounding tolerance."""
    raw = np.array([[[0, 128, 255]]], dtype=np.uint8)
    norm = normalize_optical(raw)
    assert norm.min() >= -1.0
    assert norm.max() <= 1.0
    
    recon = denormalize_optical(norm)
    np.testing.assert_allclose(recon, raw, atol=2)

def test_sar_db_normalization():
    """Sentinel-1 dB values (-35 to 0) should be mapped linearly to [-1, 1]."""
    db_vals = np.array([-35.0, -17.5, 0.0], dtype=np.float32)
    norm = normalize_sar(db_vals, min_db=-35.0, max_db=0.0)
    
    assert pytest.approx(norm[0], 0.01) == -1.0
    assert pytest.approx(norm[1], 0.01) == 0.0
    assert pytest.approx(norm[2], 0.01) == 1.0
    
    recon_db = denormalize_sar(norm, min_db=-35.0, max_db=0.0)
    np.testing.assert_allclose(recon_db, db_vals, atol=1e-3)

def test_sar_to_display_image():
    """Visual conversion should output uint8 array in [0, 255]."""
    norm = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    vis = sar_to_display_image(norm)
    assert vis.dtype == np.uint8
    assert vis[0, 0] == 0
    assert vis[0, 2] == 255

def test_split_dataset():
    """Split dataset should partition correctly without leakage."""
    dummy_pairs = [{"sample_id": f"s_{i}", "region": "r1"} for i in range(10)]
    split = split_dataset(dummy_pairs, train_ratio=0.7, val_ratio=0.3, group_by_region=False)
    assert len(split["train"]) == 7
    assert len(split["val"]) == 3
