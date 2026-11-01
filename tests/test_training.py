"""
Unit tests for training losses and checkpoint serialization.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.losses import GANLoss, Pix2PixLossComputer
from src.training.checkpoints import save_checkpoint, load_checkpoint
from src.models.pix2pix.generator import UNetGenerator
from src.models.pix2pix.discriminator import PatchGANDiscriminator

def test_gan_loss_computation():
    """GANLoss should return non-negative scalar loss."""
    gan_loss = GANLoss(use_lsgan=False)
    pred_real = torch.randn(2, 1, 30, 30)
    loss = gan_loss(pred_real, target_is_real=True)
    assert loss.dim() == 0
    assert loss.item() >= 0.0

def test_pix2pix_loss_computer():
    """Pix2PixLossComputer should compute valid generator and discriminator losses."""
    computer = Pix2PixLossComputer(lambda_l1=100.0)
    disc = PatchGANDiscriminator(in_channels=3, out_channels=1, base_filters=16)
    
    optical = torch.randn(2, 3, 64, 64)
    real_sar = torch.randn(2, 1, 64, 64)
    fake_sar = torch.randn(2, 1, 64, 64)
    
    loss_d, d_logs = computer.compute_discriminator_loss(disc, optical, real_sar, fake_sar)
    assert loss_d.dim() == 0
    assert "loss_d" in d_logs
    
    loss_g, g_logs = computer.compute_generator_loss(disc, optical, real_sar, fake_sar)
    assert loss_g.dim() == 0
    assert "loss_g_l1" in g_logs

def test_checkpoint_save_and_load(tmp_path):
    """Saving and loading checkpoint should preserve weights and metadata."""
    gen = UNetGenerator(in_channels=3, out_channels=1, base_filters=16)
    opt = torch.optim.Adam(gen.parameters(), lr=0.0002)
    
    ckpt_path = save_checkpoint(
        checkpoint_dir=tmp_path,
        experiment_id="test_exp",
        epoch=1,
        generator=gen,
        discriminator=None,
        optimizer_G=opt,
        optimizer_D=None,
        metrics={"psnr": 25.4},
        hyperparameters={"lr": 0.0002}
    )
    assert ckpt_path.exists()
    
    gen2 = UNetGenerator(in_channels=3, out_channels=1, base_filters=16)
    meta = load_checkpoint(ckpt_path, gen2)
    assert meta["epoch"] == 1
    assert meta["metrics"]["psnr"] == 25.4
