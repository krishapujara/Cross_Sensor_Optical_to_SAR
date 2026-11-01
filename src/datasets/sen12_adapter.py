"""
Dataset Adapter for SEN1-2 / SEN12MS Benchmark (Sentinel-1 and Sentinel-2).
"""
from pathlib import Path
import json
import time
import requests
from typing import Dict, Any, List, Optional
from .base_adapter import BaseDatasetAdapter
from .data_validation import validate_pair_correspondence
from ..config.settings import CACHE_DIR, HF_SEN12_BASE_URL
from ..utils.logging_config import logger
from ..utils.error_handling import retry_with_backoff

class SEN12DatasetAdapter(BaseDatasetAdapter):
    """Adapter for remote and cached Sentinel-1 / Sentinel-2 image pairs."""
    
    REGIONS = ["azerbaijan", "gabon", "germany"]
    MAX_AVAILABLE_PATCHES = 1200 # Per region in this mirror
    
    def __init__(self, cache_dir: Path = CACHE_DIR / "sen12"):
        super().__init__("SEN1-2", cache_dir)
        self.base_url = HF_SEN12_BASE_URL
        self.metadata_index_file = self.cache_dir / "metadata_index.json"
        self._ensure_metadata_index()
        
    def _ensure_metadata_index(self):
        if not self.metadata_index_file.exists():
            with open(self.metadata_index_file, "w", encoding="utf-8") as f:
                json.dump({"dataset": "SEN1-2", "samples": [], "last_updated": None}, f, indent=2)
                
    def check_availability(self) -> Dict[str, Any]:
        """Verify remote HTTP access to Sentinel-1 and Sentinel-2 endpoints."""
        test_url = f"{self.base_url}/azerbaijan/s1/patch_0000.tif"
        start_t = time.time()
        try:
            r = requests.head(test_url, allow_redirects=True, timeout=8)
            latency_ms = round((time.time() - start_t) * 1000, 1)
            is_accessible = (r.status_code == 200)
            return {
                "status": "VERIFIED ACCESSIBLE" if is_accessible else f"HTTP {r.status_code}",
                "verified_accessible": is_accessible,
                "latency_ms": latency_ms,
                "endpoint": test_url,
                "auth_required": False,
                "streaming_supported": True,
                "selective_download": True,
                "message": "Direct HTTPS access to Sentinel-1 & Sentinel-2 GeoTIFF patches is operational."
            }
        except Exception as e:
            logger.warning(f"SEN1-2 availability check failed: {e}")
            return {
                "status": "TEMPORARILY UNREACHABLE",
                "verified_accessible": False,
                "latency_ms": None,
                "endpoint": test_url,
                "auth_required": False,
                "streaming_supported": True,
                "selective_download": True,
                "message": f"Network probe failed: {str(e)}. Local cached data remains usable."
            }
            
    def get_metadata(self) -> Dict[str, Any]:
        """Return comprehensive metadata for SEN1-2."""
        return {
            "name": "SEN1-2 (SEN12MS Benchmark Mirror)",
            "sensors": {
                "SAR": "Sentinel-1 (C-band, 5.405 GHz, Interferometric Wide Swath)",
                "Optical": "Sentinel-2 MultiSpectral Instrument (MSI)"
            },
            "polarizations": ["VV", "VH (calibrated sigma-nought in dB)"],
            "optical_bands": "13 Multispectral bands (Coastal, RGB, RedEdge, NIR, SWIR)",
            "spatial_resolution": "10 m Ground Sampling Distance (GSD)",
            "dimensions": "256 x 256 pixels per patch",
            "pairing": "Spatially registered and bi-temporally matched",
            "translation_direction": "Optical-to-SAR (S2 -> S1) and SAR-to-Optical (S1 -> S2)",
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
            "source_citation": "Schmitt et al. (2019), ISPRS Annals / Technical University of Munich"
        }
        
    def list_available_samples(self, split: str = "train", limit: int = 100) -> List[Dict[str, Any]]:
        """List verifiable available remote samples."""
        samples = []
        for i in range(min(limit, self.MAX_AVAILABLE_PATCHES)):
            patch_id = f"{i:04d}"
            region = "azerbaijan" if i < 400 else "germany" if i < 800 else "gabon"
            samples.append({
                "sample_id": f"sen12_{region}_{patch_id}",
                "region": region,
                "patch_id": patch_id,
                "sar_url": f"{self.base_url}/{region}/s1/patch_{patch_id}.tif",
                "opt_url": f"{self.base_url}/{region}/s2_cloudfree/patch_{patch_id}.tif",
            })
        return samples
        
    @retry_with_backoff(retries=3, backoff_in_seconds=1.0)
    def _download_file(self, url: str, target_path: Path):
        """Download file with retry and backoff."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, stream=True, timeout=20)
        r.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                
    def retrieve_samples(self, count: int = 10, split: str = "train", progress_callback=None) -> Dict[str, Any]:
        """
        Selectively download requested number of paired samples into local cache.
        Does not download entire multi-gigabyte dataset.
        """
        available = self.list_available_samples(limit=count + 10)
        retrieved_pairs = []
        failed_count = 0
        
        # Load existing index
        with open(self.metadata_index_file, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        cached_samples = {s["sample_id"]: s for s in index_data.get("samples", [])}
        
        count_to_fetch = min(count, len(available))
        logger.info(f"Retrieving {count_to_fetch} pairs for SEN1-2...")
        
        for idx, item in enumerate(available[:count_to_fetch]):
            sample_id = item["sample_id"]
            region = item["region"]
            patch_id = item["patch_id"]
            
            sar_cache = self.cache_dir / region / "s1" / f"patch_{patch_id}.tif"
            opt_cache = self.cache_dir / region / "s2_cloudfree" / f"patch_{patch_id}.tif"
            
            try:
                # Check if already cached
                if not sar_cache.exists():
                    self._download_file(item["sar_url"], sar_cache)
                if not opt_cache.exists():
                    self._download_file(item["opt_url"], opt_cache)
                    
                val_result = self.validate_sample_pair(opt_cache, sar_cache)
                if val_result.get("valid"):
                    entry = {
                        "sample_id": sample_id,
                        "region": region,
                        "patch_id": patch_id,
                        "optical_path": str(opt_cache),
                        "sar_path": str(sar_cache),
                        "split": split,
                        "verified": True
                    }
                    cached_samples[sample_id] = entry
                    retrieved_pairs.append(entry)
                else:
                    logger.warning(f"Sample pair {sample_id} failed validation: {val_result.get('reason')}")
                    failed_count += 1
            except Exception as e:
                logger.error(f"Failed to retrieve sample {sample_id}: {e}")
                failed_count += 1
                
            if progress_callback:
                progress_callback(idx + 1, count_to_fetch)
                
        # Save updated index
        index_data["samples"] = list(cached_samples.values())
        index_data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.metadata_index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
            
        return {
            "count_retrieved": len(retrieved_pairs),
            "count_failed": failed_count,
            "cached_dir": str(self.cache_dir),
            "sample_pairs": retrieved_pairs
        }
        
    def validate_sample_pair(self, optical_path: Path, sar_path: Path) -> Dict[str, Any]:
        """Validate sample pair correspondence."""
        return validate_pair_correspondence(Path(optical_path), Path(sar_path))
        
    def get_cached_pairs(self) -> List[Dict[str, Any]]:
        """Return all successfully validated cached pairs."""
        if not self.metadata_index_file.exists():
            return []
        try:
            with open(self.metadata_index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                valid = []
                for s in data.get("samples", []):
                    if Path(s.get("optical_path", "")).exists() and Path(s.get("sar_path", "")).exists():
                        valid.append(s)
                return valid
        except Exception:
            return []
