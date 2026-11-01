"""
Unit tests for evaluation metrics (PSNR, SSIM, MAE, MSE).
"""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import (
    calculate_mse,
    calculate_mae,
    calculate_psnr,
    calculate_ssim,
    evaluate_image_pair,
    compute_difference_map
)

def test_identical_images_metrics():
    """Identical reference and generated arrays should yield MSE=0, MAE=0, high PSNR, SSIM=1."""
    img = np.random.uniform(-1.0, 1.0, (256, 256)).astype(np.float32)
    m = evaluate_image_pair(img, img, data_range=2.0)
    
    assert m["mse"] == 0.0
    assert m["mae"] == 0.0
    assert m["psnr"] >= 99.0
    assert pytest.approx(m["ssim"], 0.01) == 1.0

def test_different_images_metrics():
    """Different images should have non-zero MSE and finite PSNR."""
    img1 = np.zeros((128, 128), dtype=np.float32)
    img2 = np.ones((128, 128), dtype=np.float32)
    
    m = evaluate_image_pair(img1, img2, data_range=2.0)
    assert m["mse"] == 1.0
    assert m["mae"] == 1.0
    assert m["psnr"] > 0.0
    assert m["ssim"] < 1.0

def test_difference_map():
    """Difference map should have shape matching input and max value <= 1.0."""
    img1 = np.zeros((64, 64), dtype=np.float32)
    img2 = np.ones((64, 64), dtype=np.float32)
    diff = compute_difference_map(img1, img2)
    assert diff.shape == (64, 64)
    assert np.max(diff) <= 1.0
