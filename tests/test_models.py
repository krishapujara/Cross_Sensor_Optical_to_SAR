"""
Unit tests for model architectures (UNetGenerator, PatchGANDiscriminator, ResNetGenerator).
"""
import torch
import pytest
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.pix2pix.generator import UNetGenerator
from src.models.pix2pix.discriminator import PatchGANDiscriminator
from src.models.cyclegan.generator import ResNetGenerator
from src.models.cyclegan.discriminator import CycleGANDiscriminator
from src.models.model_registry import build_pix2pix_models

def test_unet_generator_shape():
    """UNetGenerator forward pass on (B, 3, 256, 256) should yield (B, 1, 256, 256) in [-1, 1]."""
    gen = UNetGenerator(in_channels=3, out_channels=1, base_filters=16) # Smaller base_filters for quick test
    dummy_input = torch.randn(2, 3, 256, 256)
    output = gen(dummy_input)
    
    assert output.shape == (2, 1, 256, 256)
    assert output.min().item() >= -1.0
    assert output.max().item() <= 1.0

def test_patchgan_discriminator_shape():
    """PatchGANDiscriminator should output 2D spatial patch logits."""
    disc = PatchGANDiscriminator(in_channels=3, out_channels=1, base_filters=16)
    opt_in = torch.randn(2, 3, 256, 256)
    sar_in = torch.randn(2, 1, 256, 256)
    
    logits = disc(opt_in, sar_in)
    # Output shape should be (B, 1, H_out, W_out)
    assert logits.shape[0] == 2
    assert logits.shape[1] == 1
    assert logits.dim() == 4

def test_cyclegan_models_shape():
    """CycleGAN generator and discriminator forward test."""
    gen = ResNetGenerator(in_channels=3, out_channels=1, num_blocks=2, base_filters=16)
    disc = CycleGANDiscriminator(in_channels=1, base_filters=16)
    
    dummy_in = torch.randn(1, 3, 128, 128)
    fake_sar = gen(dummy_in)
    assert fake_sar.shape == (1, 1, 128, 128)
    
    d_out = disc(fake_sar)
    assert d_out.dim() == 4

def test_build_pix2pix_models():
    """Test model registry factory function."""
    gen, disc = build_pix2pix_models(in_channels=3, out_channels=1, base_filters=16)
    assert isinstance(gen, UNetGenerator)
    assert isinstance(disc, PatchGANDiscriminator)
