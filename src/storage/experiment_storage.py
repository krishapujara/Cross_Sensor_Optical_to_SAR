"""
Persistent JSON storage for tracking machine learning experiment runs.
"""
from pathlib import Path
import json
import time
from typing import Dict, Any, List, Optional
from ..config.settings import EXPERIMENTS_DIR
from ..utils.logging_config import logger

class ExperimentStorage:
    """Manages experiment history records stored in structured JSON."""
    
    def __init__(self, storage_dir: Path = EXPERIMENTS_DIR):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.storage_dir / "experiments.json"
        self._ensure_file()
        
    def _ensure_file(self):
        if not self.history_file.exists():
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)
                
    def list_experiments(self) -> List[Dict[str, Any]]:
        """Retrieve all recorded experiments."""
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading experiments: {e}")
            return []
            
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve experiment by ID."""
        for exp in self.list_experiments():
            if exp.get("experiment_id") == experiment_id:
                return exp
        return None
        
    def save_experiment(self, experiment_data: Dict[str, Any]) -> str:
        """Save or update an experiment record."""
        experiments = self.list_experiments()
        exp_id = experiment_data.get("experiment_id") or f"EXP-{int(time.time())}"
        experiment_data["experiment_id"] = exp_id
        experiment_data["timestamp"] = experiment_data.get("timestamp") or time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if already exists
        existing_idx = next((i for i, x in enumerate(experiments) if x.get("experiment_id") == exp_id), None)
        if existing_idx is not None:
            experiments[existing_idx] = experiment_data
        else:
            experiments.insert(0, experiment_data)
            
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(experiments, f, indent=2)
            
        logger.info(f"Saved experiment {exp_id}")
        return exp_id
        
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment record."""
        experiments = self.list_experiments()
        filtered = [x for x in experiments if x.get("experiment_id") != experiment_id]
        if len(filtered) != len(experiments):
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(filtered, f, indent=2)
            return True
        return False
