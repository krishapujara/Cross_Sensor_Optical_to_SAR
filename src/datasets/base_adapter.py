"""
Abstract Base Class defining the standard interface for all remote and local dataset adapters.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path

class BaseDatasetAdapter(ABC):
    """Abstract interface for satellite image translation dataset adapters."""
    
    def __init__(self, name: str, cache_dir: Path):
        self.name = name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def check_availability(self) -> Dict[str, Any]:
        """
        Probe remote or local source availability.
        Returns:
            Dict containing status, latency_ms, verified_accessible (bool),
            message, and authentication requirement status.
        """
        pass
        
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return structured dataset metadata (sensor, resolution, licensing,
        channels, pairing, supported translation direction).
        """
        pass
        
    @abstractmethod
    def list_available_samples(self, split: str = "train", limit: int = 100) -> List[Dict[str, Any]]:
        """List verifiable available sample identifiers or URLs without downloading payloads."""
        pass
        
    @abstractmethod
    def retrieve_samples(self, count: int, split: str = "train", progress_callback=None) -> Dict[str, Any]:
        """
        Selectively download or stream the requested number of paired samples into local cache.
        Returns:
            Dict containing count_retrieved, count_failed, cached_dir, sample_pairs.
        """
        pass
        
    @abstractmethod
    def validate_sample_pair(self, optical_path: Path, sar_path: Path) -> Dict[str, Any]:
        """Validate integrity, shape, and channels of a single paired sample."""
        pass
