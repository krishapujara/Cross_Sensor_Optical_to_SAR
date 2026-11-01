"""
Batch evaluation framework across test datasets and external benchmarks.
"""
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
from .metrics import evaluate_image_pair
from ..preprocessing.image_loader import load_optical_image, load_sar_image
from ..preprocessing.normalization import sar_to_display_image, denormalize_optical
from ..models.pix2pix.generator import UNetGenerator
from ..training.checkpoints import load_checkpoint
from ..utils.logging_config import logger

class ModelEvaluator:
    """Evaluates trained generator model on held-out test datasets or individual pairs."""
    
    def __init__(self, checkpoint_path: Path, device: torch.device = torch.device("cpu")):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        
        # Check base_filters in checkpoint
        ckpt_data = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        base_filters = ckpt_data.get("hyperparameters", {}).get("base_filters", 64)
        
        self.generator = UNetGenerator(in_channels=3, out_channels=1, base_filters=base_filters).to(self.device)
        self.meta = load_checkpoint(self.checkpoint_path, self.generator, device=self.device)
        self.generator.eval()
        
    def evaluate_pairs(self, pairs: List[Dict[str, Any]], img_size: int = 256) -> Dict[str, Any]:
        """
        Evaluate model across a list of test sample pairs.
        Returns aggregate metrics, standard deviations, and individual pair evaluations.
        """
        if not pairs:
            return {"count": 0, "metrics": {}}
            
        psnr_vals, ssim_vals, mae_vals, mse_vals = [], [], [], []
        sample_results = []
        
        start_time = time.time()
        self.generator.eval()
        
        with torch.no_grad():
            for idx, p in enumerate(pairs):
                opt_path = Path(p["optical_path"])
                sar_path = Path(p["sar_path"])
                
                try:
                    opt_norm, _ = load_optical_image(opt_path, target_size=(img_size, img_size))
                    sar_norm, _ = load_sar_image(sar_path, target_size=(img_size, img_size))
                    
                    tensor_in = torch.from_numpy(opt_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
                    tensor_pred = self.generator(tensor_in)
                    
                    fake_sar_norm = tensor_pred.squeeze().cpu().numpy()
                    real_sar_norm = sar_norm.squeeze()
                    
                    m = evaluate_image_pair(real_sar_norm, fake_sar_norm)
                    psnr_vals.append(m["psnr"])
                    ssim_vals.append(m["ssim"])
                    mae_vals.append(m["mae"])
                    mse_vals.append(m["mse"])
                    
                    sample_results.append({
                        "sample_id": p.get("sample_id", f"sample_{idx}"),
                        "optical_path": str(opt_path),
                        "sar_path": str(sar_path),
                        "optical_visual": denormalize_optical(opt_norm),
                        "reference_sar_visual": sar_to_display_image(real_sar_norm),
                        "synthetic_sar_visual": sar_to_display_image(fake_sar_norm),
                        "metrics": m
                    })
                except Exception as e:
                    logger.warning(f"Error evaluating pair {p.get('sample_id')}: {e}")
                    
        eval_time = round(time.time() - start_time, 2)
        
        agg_metrics = {
            "psnr_mean": round(float(np.mean(psnr_vals)), 3) if psnr_vals else 0.0,
            "psnr_std": round(float(np.std(psnr_vals)), 3) if psnr_vals else 0.0,
            "ssim_mean": round(float(np.mean(ssim_vals)), 4) if ssim_vals else 0.0,
            "ssim_std": round(float(np.std(ssim_vals)), 4) if ssim_vals else 0.0,
            "mae_mean": round(float(np.mean(mae_vals)), 4) if mae_vals else 0.0,
            "mse_mean": round(float(np.mean(mse_vals)), 4) if mse_vals else 0.0,
        }
        
        return {
            "checkpoint": self.checkpoint_path.name,
            "sample_count": len(sample_results),
            "aggregate_metrics": agg_metrics,
            "evaluation_time_sec": eval_time,
            "samples": sample_results
        }
