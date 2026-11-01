"""
End-to-end inference engine for optical-to-SAR satellite image translation.
"""
from pathlib import Path
import time
import numpy as np
import torch
from typing import Dict, Any, Union, Optional
from ..models.pix2pix.generator import UNetGenerator
from ..training.checkpoints import load_checkpoint
from ..preprocessing.image_loader import load_optical_image
from ..preprocessing.normalization import denormalize_optical
from ..preprocessing.geospatial_processing import extract_geospatial_metadata
from .output_processor import process_sar_prediction, export_as_png_bytes, export_as_geotiff_bytes
from ..utils.device import get_torch_device
from ..utils.logging_config import logger

class OpticalToSARPredictor:
    """Predictor translating single optical inputs into synthetic SAR imagery."""
    
    def __init__(self, checkpoint_path: Optional[Path] = None, base_filters: int = 64, device: Optional[torch.device] = None):
        self.device = device or get_torch_device()
        self.base_filters = base_filters
        self.generator = UNetGenerator(in_channels=3, out_channels=1, base_filters=self.base_filters).to(self.device)
        self.checkpoint_metadata = {}
        self.checkpoint_path = None
        
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_model(Path(checkpoint_path))
            
    def load_model(self, checkpoint_path: Path):
        """Load checkpoint weights into generator."""
        self.checkpoint_path = Path(checkpoint_path)
        # Check if checkpoint specified base_filters
        ckpt_data = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        ckpt_filters = ckpt_data.get("hyperparameters", {}).get("base_filters")
        if ckpt_filters and ckpt_filters != self.base_filters:
            self.base_filters = ckpt_filters
            self.generator = UNetGenerator(in_channels=3, out_channels=1, base_filters=self.base_filters).to(self.device)
            
        self.checkpoint_metadata = load_checkpoint(self.checkpoint_path, self.generator, device=self.device)
        self.generator.eval()
        logger.info(f"Predictor ready with weights from {self.checkpoint_path.name}")
        
    def predict(self, input_image: Union[Path, str, bytes], img_size: int = 256) -> Dict[str, Any]:
        """
        Execute cross-sensor translation on single optical input.
        Returns visual optical input, synthetic SAR, calibrated dB array, timing, and metadata.
        """
        start_time = time.time()
        
        # 1. Preprocess input
        if isinstance(input_image, (str, Path)):
            input_path = Path(input_image)
            norm_opt, opt_meta = load_optical_image(input_path, target_size=(img_size, img_size))
            geo_meta = extract_geospatial_metadata(input_path)
        else:
            # Handle uploaded bytes
            import io
            from PIL import Image
            with Image.open(io.BytesIO(input_image)) as im:
                im = im.convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
                raw_np = np.array(im)
            norm_opt = (raw_np.astype(np.float32) / 127.5) - 1.0
            opt_meta = {"filename": "uploaded_optical.png", "format": "PNG", "raw_shape": raw_np.shape}
            geo_meta = {"is_geospatial": False}
            
        # 2. PyTorch tensor conversion: (H, W, C) -> (1, C, H, W)
        tensor_in = torch.from_numpy(norm_opt.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        
        # 3. Model Inference
        self.generator.eval()
        with torch.no_grad():
            tensor_out = self.generator(tensor_in)
            
        inference_time_ms = round((time.time() - start_time) * 1000, 1)
        
        # 4. Format outputs: (1, 1, H, W) -> (H, W)
        raw_pred_norm = tensor_out.squeeze().cpu().numpy()
        processed = process_sar_prediction(raw_pred_norm)
        
        # Visual Optical image for comparison: denormalized uint8 RGB
        opt_visual = denormalize_optical(norm_opt)
        
        return {
            "optical_visual": opt_visual,
            "synthetic_sar_visual": processed["visual_image"],
            "calibrated_db": processed["calibrated_db"],
            "stats": processed["stats"],
            "inference_time_ms": inference_time_ms,
            "input_metadata": opt_meta,
            "geospatial_metadata": geo_meta,
            "png_bytes": export_as_png_bytes(processed["visual_image"]),
            "geotiff_bytes": export_as_geotiff_bytes(processed["calibrated_db"]),
            "model_checkpoint": self.checkpoint_path.name if self.checkpoint_path else "Uninitialized",
            "device": str(self.device)
        }
