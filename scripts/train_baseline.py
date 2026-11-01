"""
Train a fast, high quality Pix2Pix baseline model on verified normalized SEN1-2 pairs.
"""
from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.sen12_adapter import SEN12DatasetAdapter
from src.preprocessing.patch_extraction import split_dataset, create_dataloaders
from src.models.model_registry import build_pix2pix_models
from src.training.trainer import Pix2PixTrainer
from src.config.settings import CHECKPOINTS_DIR

def run():
    adapter = SEN12DatasetAdapter()
    pairs = adapter.get_cached_pairs()
    print(f"Loaded {len(pairs)} verified pairs for baseline training.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    splits = split_dataset(pairs, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0, group_by_region=False)
    loaders = create_dataloaders(splits, batch_size=2, img_size=256)
    
    gen, disc = build_pix2pix_models(in_channels=3, out_channels=1, base_filters=32, device=device)
    
    trainer = Pix2PixTrainer(
        generator=gen,
        discriminator=disc,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        experiment_id="pix2pix_sentinel_baseline",
        lr=0.0002,
        lambda_l1=100.0,
        checkpoint_dir=CHECKPOINTS_DIR,
        device=device
    )
    
    print("Beginning training 2 epochs...")
    history = trainer.train(epochs=2)
    print("Training successfully completed!")
    for h in history:
        print(f"Epoch {h['epoch']}: Loss G={h['loss_g']:.4f}, Loss D={h['loss_d']:.4f}, Val PSNR={h['val_psnr']:.2f} dB, Val SSIM={h['val_ssim']:.4f}")

if __name__ == "__main__":
    run()
