"""
Local cache management for remote dataset samples.
"""
from pathlib import Path
import shutil
import json
from typing import Dict, Any, List
from ..config.settings import CACHE_DIR
from ..utils.logging_config import logger

class CacheManager:
    """Manages cached dataset image pairs on local disk."""
    
    def __init__(self, base_cache_dir: Path = CACHE_DIR):
        self.base_cache_dir = Path(base_cache_dir)
        self.base_cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_dataset_cache_dir(self, dataset_name: str) -> Path:
        """Return dedicated cache path for a dataset."""
        d_path = self.base_cache_dir / dataset_name.lower().replace("-", "_")
        d_path.mkdir(parents=True, exist_ok=True)
        return d_path
        
    def get_cache_stats(self, dataset_name: str = None) -> Dict[str, Any]:
        """Compute cache usage statistics."""
        target_dir = self.get_dataset_cache_dir(dataset_name) if dataset_name else self.base_cache_dir
        if not target_dir.exists():
            return {"file_count": 0, "size_mb": 0.0, "pairs_count": 0}
            
        total_size = 0
        file_count = 0
        pairs = 0
        
        for p in target_dir.rglob("*"):
            if p.is_file():
                file_count += 1
                total_size += p.stat().st_size
                
        # Count pairs based on metadata index if present
        meta_file = target_dir / "metadata_index.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                    pairs = len(meta_data.get("samples", []))
            except Exception:
                pairs = file_count // 2
        else:
            pairs = file_count // 2
            
        return {
            "file_count": file_count,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "size_bytes": total_size,
            "pairs_count": pairs,
            "path": str(target_dir)
        }
        
    def clear_cache(self, dataset_name: str = None) -> bool:
        """Clear cache for a dataset or entire cache directory."""
        try:
            target_dir = self.get_dataset_cache_dir(dataset_name) if dataset_name else self.base_cache_dir
            if target_dir.exists():
                for item in target_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            logger.info(f"Cleared cache at {target_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
