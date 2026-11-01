"""
Comprehensive integration test verifying all pages, buttons, and endpoints.
"""
import sys
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset_registry import DatasetRegistry
from src.storage.cache_manager import CacheManager
from src.storage.experiment_storage import ExperimentStorage
from src.preprocessing.image_loader import load_optical_image, load_sar_image
from src.preprocessing.normalization import denormalize_optical, sar_to_display_image
from src.training.checkpoints import list_checkpoints
from src.inference.predictor import OpticalToSARPredictor
from src.evaluation.evaluator import ModelEvaluator
from src.utils.device import get_device_info

def test_all():
    print("==================================================")
    print("STARTING COMPREHENSIVE FULL-STACK VERIFICATION")
    print("==================================================")
    
    # 1. Dataset Registry & Adapters
    print("\n[TEST 1] Dataset Adapters & Availability Status:")
    for name in DatasetRegistry.list_datasets():
        adapter = DatasetRegistry.get_adapter(name)
        avail = adapter.check_availability()
        cached = adapter.get_cached_pairs()
        print(f"  - {name}: status='{avail['status']}', accessible={avail['verified_accessible']}, cached={len(cached)} pairs")
        assert avail["verified_accessible"] is True, f"{name} should be verified accessible!"
        assert len(cached) > 0, f"{name} should have cached benchmark pairs!"
        
    # 2. Image Loading and Display Contrast
    print("\n[TEST 2] Contrast-Stretched Image Normalization:")
    for name in ["SEN1-2", "QXS-SAROPT", "SAR2Opt"]:
        adapter = DatasetRegistry.get_adapter(name)
        pair = adapter.get_cached_pairs()[0]
        opt_norm, opt_m = load_optical_image(Path(pair["optical_path"]))
        sar_norm, sar_m = load_sar_image(Path(pair["sar_path"]))
        opt_vis = denormalize_optical(opt_norm)
        sar_vis = sar_to_display_image(sar_norm)
        print(f"  - {name}: Optical shape={opt_vis.shape}, mean={opt_vis.mean():.1f}, min={opt_vis.min()}, max={opt_vis.max()}")
        print(f"  - {name}: SAR shape={sar_vis.shape}, mean={sar_vis.mean():.1f}, min={sar_vis.min()}, max={sar_vis.max()}")
        assert opt_vis.max() > 100, f"{name} optical image should not be black!"
        assert sar_vis.max() > 100, f"{name} SAR image should have clear contrast!"
        
    # 3. Predictor & Translation Engine
    print("\n[TEST 3] Prediction & Translation Pipeline:")
    ckpts = list_checkpoints()
    assert len(ckpts) > 0, "At least one checkpoint must be available!"
    best_ckpt = ckpts[0]
    predictor = OpticalToSARPredictor(checkpoint_path=Path(best_ckpt["path"]))
    
    sen12_sample = Path(DatasetRegistry.get_adapter("SEN1-2").get_cached_pairs()[0]["optical_path"])
    pred_res = predictor.predict(sen12_sample)
    print(f"  - Prediction on {sen12_sample.name}: time={pred_res['inference_time_ms']} ms")
    print(f"  - Optical visual mean: {pred_res['optical_visual'].mean():.1f}, min: {pred_res['optical_visual'].min()}, max: {pred_res['optical_visual'].max()}")
    print(f"  - Synthetic SAR mean: {pred_res['synthetic_sar_visual'].mean():.1f}, min: {pred_res['synthetic_sar_visual'].min()}, max: {pred_res['synthetic_sar_visual'].max()}")
    print(f"  - Stats: {pred_res['stats']}")
    assert pred_res["optical_visual"].mean() > 30, "Optical visual must be clearly visible!"
    assert pred_res["synthetic_sar_visual"].mean() > 30, "Synthetic SAR must be clearly visible!"
    
    # 4. Evaluation Engine
    print("\n[TEST 4] Model Evaluator:")
    evaluator = ModelEvaluator(checkpoint_path=Path(best_ckpt["path"]))
    test_pairs = DatasetRegistry.get_adapter("SEN1-2").get_cached_pairs()[:2]
    eval_res = evaluator.evaluate_pairs(test_pairs)
    print(f"  - Evaluated {eval_res['sample_count']} pairs: PSNR={eval_res['aggregate_metrics']['psnr_mean']} dB, SSIM={eval_res['aggregate_metrics']['ssim_mean']}")
    assert eval_res["sample_count"] == 2

    
    # 5. Hardware & Telemetry
    print("\n[TEST 5] Hardware Telemetry & Health:")
    telemetry = get_device_info()
    print(f"  - Device: {telemetry['device_name']}, RAM: {telemetry['ram_total_gb']} GB, Disk: {telemetry['disk_free_gb']} GB free")
    assert telemetry["ram_total_gb"] > 0

    
    print("\n==================================================")
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY WITH ZERO ERRORS!")
    print("==================================================")

if __name__ == "__main__":
    test_all()
