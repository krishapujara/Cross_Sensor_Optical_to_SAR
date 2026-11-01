"""
Robust checkpoint serialization and restoration with reproducible metadata.
"""
from pathlib import Path
import json
import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ..config.settings import CHECKPOINTS_DIR
from ..utils.logging_config import logger

def save_checkpoint(
    checkpoint_dir: Path,
    experiment_id: str,
    epoch: int,
    generator: nn.Module,
    discriminator: Optional[nn.Module],
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: Optional[torch.optim.Optimizer],
    metrics: Dict[str, float],
    hyperparameters: Dict[str, Any],
    is_best: bool = False
) -> Path:
    """Save PyTorch model weights and training metadata."""
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{experiment_id}_epoch_{epoch}.pt"
    save_path = ckpt_dir / filename
    
    payload = {
        "experiment_id": experiment_id,
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict() if discriminator else None,
        "optimizer_G_state_dict": optimizer_G.state_dict() if optimizer_G else None,
        "optimizer_D_state_dict": optimizer_D.state_dict() if optimizer_D else None,
        "metrics": metrics,
        "hyperparameters": hyperparameters,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__
    }
    
    torch.save(payload, save_path)
    logger.info(f"Saved checkpoint to {save_path}")
    
    if is_best:
        best_path = ckpt_dir / f"{experiment_id}_best.pt"
        torch.save(payload, best_path)
        
    # Write metadata index
    meta_path = ckpt_dir / f"{filename}.json"
    meta_json = {
        "filename": filename,
        "experiment_id": experiment_id,
        "epoch": epoch,
        "metrics": metrics,
        "hyperparameters": hyperparameters,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2)
        
    return save_path

def load_checkpoint(
    checkpoint_path: Path,
    generator: nn.Module,
    discriminator: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """Load model weights from checkpoint into existing generator and discriminator instances."""
    p = Path(checkpoint_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {p}")
        
    checkpoint = torch.load(p, map_location=device, weights_only=False)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    if discriminator and checkpoint.get("discriminator_state_dict"):
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        
    logger.info(f"Loaded checkpoint from {p} (Epoch {checkpoint.get('epoch')})")
    return {
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics", {}),
        "hyperparameters": checkpoint.get("hyperparameters", {}),
        "timestamp": checkpoint.get("timestamp")
    }

def list_checkpoints(checkpoint_dir: Path = CHECKPOINTS_DIR) -> List[Dict[str, Any]]:
    """List all available saved checkpoint files."""
    d = Path(checkpoint_dir)
    ckpts = []
    for f in sorted(d.glob("*.pt"), reverse=True):
        meta_file = d / f"{f.name}.json"
        meta = {}
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as m:
                    meta = json.load(m)
            except Exception:
                pass
        ckpts.append({
            "path": str(f),
            "filename": f.name,
            "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            "epoch": meta.get("epoch", "Unknown"),
            "metrics": meta.get("metrics", {}),
            "created_at": meta.get("saved_at", time.ctime(f.stat().st_mtime))
        })
    return ckpts
