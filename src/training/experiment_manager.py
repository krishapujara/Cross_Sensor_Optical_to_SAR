"""
Experiment management tying training runs to persistent storage.
"""
from pathlib import Path
from typing import Dict, Any
from .checkpoints import list_checkpoints
from ..storage.experiment_storage import ExperimentStorage
from ..utils.logging_config import logger

class ExperimentManager:
    """Manages experiment records and checkpoint linkages."""
    def __init__(self):
        self.storage = ExperimentStorage()
        
    def record_training_run(self, run_results: Dict[str, Any], dataset_name: str, split_info: Dict[str, Any]) -> str:
        """Create structured experiment record from completed training session."""
        exp_id = run_results.get("experiment_id")
        final_m = run_results.get("final_metrics", {})
        
        record = {
            "experiment_id": exp_id,
            "dataset": dataset_name,
            "model_architecture": "Pix2Pix (UNet-256 + PatchGAN-70)",
            "epochs_trained": run_results.get("epochs_completed", 0),
            "total_duration_sec": run_results.get("total_duration_sec", 0.0),
            "best_checkpoint": run_results.get("best_checkpoint"),
            "best_val_psnr": run_results.get("best_val_psnr", 0.0),
            "final_metrics": {
                "loss_g": final_m.get("loss_g", 0.0),
                "loss_d": final_m.get("loss_d", 0.0),
                "psnr": final_m.get("val_psnr", 0.0),
                "ssim": final_m.get("val_ssim", 0.0),
                "mae": final_m.get("val_mae", 0.0),
                "mse": final_m.get("val_mse", 0.0)
            },
            "split_info": {
                "train_count": len(split_info.get("train", [])),
                "val_count": len(split_info.get("val", [])),
                "test_count": len(split_info.get("test", []))
            },
            "history": run_results.get("history", [])
        }
        self.storage.save_experiment(record)
        return exp_id
        
    def get_latest_experiment(self) -> Dict[str, Any]:
        """Return the most recent experiment record."""
        exps = self.storage.list_experiments()
        return exps[0] if exps else None
