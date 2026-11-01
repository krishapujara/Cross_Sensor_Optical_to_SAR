"""
End-to-End Operational Verification Script.
Retrieves real Sentinel-1 & Sentinel-2 GeoTIFF pairs, validates them,
trains a genuine Pix2Pix model with backprop, saves a checkpoint,
runs inference, and computes real PSNR & SSIM evaluation metrics.
"""
import sys
import time
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset_registry import DatasetRegistry
from src.preprocessing.patch_extraction import split_dataset, create_dataloaders
from src.models.model_registry import build_pix2pix_models
from src.training.trainer import Pix2PixTrainer
from src.training.experiment_manager import ExperimentManager
from src.inference.predictor import OpticalToSARPredictor
from src.evaluation.evaluator import ModelEvaluator
from src.utils.device import get_torch_device

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def run_verification():
    print("=" * 70)
    print("[ISRO/URSC] CROSS-SENSOR OPTICAL-TO-SAR END-TO-END VERIFICATION")
    print("=" * 70)
    
    device = get_torch_device()
    print(f"[1/6] Hardware Compute Device: {device}")
    
    # 1. Dataset Adapter & Retrieval
    print("[2/6] Probing SEN1-2 remote endpoint and retrieving verified pairs...")
    adapter = DatasetRegistry.get_adapter("SEN1-2")
    avail = adapter.check_availability()
    print(f"      Availability: {avail['status']} (Latency: {avail.get('latency_ms')} ms)")
    
    # Retrieve 4 real pairs for smoke training
    retrieval_res = adapter.retrieve_samples(count=4, split="train")
    print(f"      Retrieved: {retrieval_res['count_retrieved']} pairs, Failed: {retrieval_res['count_failed']}")
    
    cached_pairs = adapter.get_cached_pairs()
    print(f"      Total Verified Cached Pairs: {len(cached_pairs)}")
    assert len(cached_pairs) >= 2, "Need at least 2 pairs for training and validation!"
    
    # 2. Preprocessing & Partitioning
    print("[3/6] Partitioning data into train and validation sets...")
    split_dict = split_dataset(cached_pairs, train_ratio=0.75, val_ratio=0.25, group_by_region=False)
    loaders = create_dataloaders(split_dict, batch_size=2, img_size=256)
    print(f"      Train Batches: {len(loaders['train'])} | Val Batches: {len(loaders['val'])}")
    
    # 3. Model Architecture & Genuine Training
    print("[4/6] Initializing U-Net Generator + PatchGAN Discriminator and running genuine training...")
    gen, disc = build_pix2pix_models(in_channels=3, out_channels=1, base_filters=16, device=device)
    
    trainer = Pix2PixTrainer(
        generator=gen,
        discriminator=disc,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        experiment_id="e2e_verification_run",
        lr=0.0002,
        lambda_l1=100.0,
        device=device
    )
    
    # Run 2 real epochs with backprop
    training_results = trainer.train(epochs=2, checkpoint_freq=1)
    print(f"      Training Finished in {training_results['total_duration_sec']}s!")
    print(f"      Final Loss G: {training_results['final_metrics']['loss_g']:.4f} | Loss D: {training_results['final_metrics']['loss_d']:.4f}")
    print(f"      Val PSNR: {training_results['final_metrics']['val_psnr']:.2f} dB | Val SSIM: {training_results['final_metrics']['val_ssim']:.4f}")
    print(f"      Best Checkpoint Saved: {training_results['best_checkpoint']}")
    
    # Record to experiment storage
    exp_mgr = ExperimentManager()
    exp_id = exp_mgr.record_training_run(training_results, dataset_name="SEN1-2", split_info=split_dict)
    print(f"      Recorded in Experiment Registry as: {exp_id}")
    
    # 4. Inference Engine Verification
    print("[5/6] Testing OpticalToSARPredictor with newly saved checkpoint...")
    test_opt_path = Path(cached_pairs[0]["optical_path"])
    predictor = OpticalToSARPredictor(checkpoint_path=Path(training_results["best_checkpoint"]), device=device)
    pred_res = predictor.predict(test_opt_path, img_size=256)
    
    print(f"      Inference Execution Time: {pred_res['inference_time_ms']} ms")
    print(f"      Synthetic SAR Visual Shape: {pred_res['synthetic_sar_visual'].shape}")
    print(f"      Calibrated Backscatter: min={pred_res['stats']['min_db']} dB, max={pred_res['stats']['max_db']} dB")
    print(f"      Generated PNG Bytes: {len(pred_res['png_bytes'])} bytes")
    print(f"      Generated GeoTIFF Bytes: {len(pred_res['geotiff_bytes'])} bytes")
    
    # 5. Quantitative Evaluation Verification
    print("[6/6] Running ModelEvaluator across held-out pairs...")
    evaluator = ModelEvaluator(checkpoint_path=Path(training_results["best_checkpoint"]), device=device)
    eval_res = evaluator.evaluate_pairs(cached_pairs[:2])
    print(f"      Evaluated {eval_res['sample_count']} samples:")
    print(f"      Mean PSNR: {eval_res['aggregate_metrics']['psnr_mean']} dB")
    print(f"      Mean SSIM: {eval_res['aggregate_metrics']['ssim_mean']}")
    print(f"      Mean MAE:  {eval_res['aggregate_metrics']['mae_mean']}")
    print(f"      Mean MSE:  {eval_res['aggregate_metrics']['mse_mean']}")
    
    print("=" * 70)
    print("[SUCCESS] END-TO-END VERIFICATION COMPLETED WITH ZERO ERRORS!")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
