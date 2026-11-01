"""
Unit tests for inference pipeline and output post-processing.
"""
import pytest
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.output_processor import (
    process_sar_prediction,
    export_as_png_bytes,
    export_as_geotiff_bytes
)
from src.inference.predictor import OpticalToSARPredictor
from src.models.pix2pix.generator import UNetGenerator
from src.training.checkpoints import save_checkpoint

def test_output_processor():
    """Synthetic SAR raw array should be converted to visual image and calibrated dB values."""
    raw_norm = np.random.uniform(-1.0, 1.0, (128, 128)).astype(np.float32)
    processed = process_sar_prediction(raw_norm)
    
    assert processed["visual_image"].dtype == np.uint8
    assert processed["visual_image"].shape == (128, 128)
    assert "min_db" in processed["stats"]
    
    png_bytes = export_as_png_bytes(processed["visual_image"])
    assert len(png_bytes) > 0
    
    geotiff_bytes = export_as_geotiff_bytes(processed["calibrated_db"])
    assert len(geotiff_bytes) > 0

def test_predictor_forward_pass(tmp_path):
    """Predictor should accept image file and return synthetic SAR product."""
    test_img = tmp_path / "test_optical.png"
    Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)).save(test_img)
    
    gen = UNetGenerator(in_channels=3, out_channels=1, base_filters=16)
    ckpt_path = save_checkpoint(
        checkpoint_dir=tmp_path,
        experiment_id="inf_test",
        epoch=1,
        generator=gen,
        discriminator=None,
        optimizer_G=None,
        optimizer_D=None,
        metrics={},
        hyperparameters={"base_filters": 16}
    )
    
    predictor = OpticalToSARPredictor(checkpoint_path=ckpt_path)
    res = predictor.predict(test_img, img_size=256)
    
    assert "optical_visual" in res
    assert "synthetic_sar_visual" in res
    assert "calibrated_db" in res
    assert res["inference_time_ms"] > 0
