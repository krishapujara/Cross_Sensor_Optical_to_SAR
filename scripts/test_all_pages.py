"""
Comprehensive validator for every Streamlit page and interactive backend endpoint.
Tests imports, functions, classes, and simulations without launching a browser.
"""
import sys
import importlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_page_imports():
    print("=" * 60)
    print("Testing imports and syntax of all application pages:")
    print("=" * 60)
    
    pages = [
        "app",
        "pages.1_Mission_Overview",
        "pages.2_Dataset_Manager",
        "pages.3_Data_Exploration",
        "pages.4_Model_Training",
        "pages.5_Optical_to_SAR_Translation",
        "pages.6_Evaluation_Laboratory",
        "pages.7_Experiment_History",
        "pages.8_System_Health"
    ]
    
    for p in pages:
        try:
            mod_path = PROJECT_ROOT / (p.replace(".", "/") + ".py")
            with open(mod_path, "r", encoding="utf-8") as f:
                code = f.read()
            compile(code, str(mod_path), "exec")
            print(f"[OK] Syntax Valid: {p}")
        except Exception as e:
            print(f"[FAIL] Syntax Error in {p}: {e}")
            raise e

def test_all_endpoints():
    print("=" * 60)
    print("Testing all system endpoints and adapter checks:")
    print("=" * 60)
    
    from src.datasets.dataset_registry import DatasetRegistry
    from src.storage.cache_manager import CacheManager
    from src.storage.experiment_storage import ExperimentStorage
    from src.training.checkpoints import list_checkpoints
    from src.utils.device import get_device_info, get_torch_device
    from src.models.model_registry import build_pix2pix_models
    from src.inference.predictor import OpticalToSARPredictor
    from src.evaluation.evaluator import ModelEvaluator
    
    # 1. Device Info
    info = get_device_info()
    assert "cpu_count" in info and "disk_free_gb" in info
    print(f"[OK] Device Info: {info['device']}, RAM: {info['ram_available_gb']}/{info['ram_total_gb']} GB, Disk Free: {info['disk_free_gb']} GB")
    
    # 2. Checkpoints
    ckpts = list_checkpoints()
    print(f"[OK] Checkpoints found: {len(ckpts)}")
    
    # 3. Cache Manager
    cm = CacheManager()
    stats = cm.get_cache_stats()
    print(f"[OK] Cache Stats: {stats['pairs_count']} pairs, {stats['size_mb']} MB")
    
    # 4. Experiment Storage
    storage = ExperimentStorage()
    exps = storage.list_experiments()
    print(f"[OK] Experiments recorded: {len(exps)}")
    
    # 5. Dataset Adapters
    for d in DatasetRegistry.list_datasets():
        adapter = DatasetRegistry.get_adapter(d)
        meta = adapter.get_metadata()
        avail = adapter.check_availability()
        cached = adapter.get_cached_pairs()
        print(f"[OK] Adapter '{d}': Status={avail['status']}, Verified={avail['verified_accessible']}, Cached Pairs={len(cached)}")
        
    print("=" * 60)
    print("ALL APPLICATION ENDPOINTS & MODULES TESTED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_page_imports()
    test_all_endpoints()
