"""
Factory registry and weight initialization for cross-sensor translation models.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .pix2pix.generator import UNetGenerator
from .pix2pix.discriminator import PatchGANDiscriminator
from .cyclegan.generator import ResNetGenerator
from .cyclegan.discriminator import CycleGANDiscriminator
from ..utils.logging_config import logger

def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02):
    """Initialize network weights as in original Pix2Pix paper (Gaussian mean=0, std=0.02)."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, "weight") and classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

def build_pix2pix_models(
    in_channels: int = 3,
    out_channels: int = 1,
    base_filters: int = 64,
    device: torch.device = torch.device("cpu")
) -> Tuple[UNetGenerator, PatchGANDiscriminator]:
    """Instantiate and initialize Pix2Pix Generator and Discriminator."""
    gen = UNetGenerator(in_channels=in_channels, out_channels=out_channels, base_filters=base_filters)
    disc = PatchGANDiscriminator(in_channels=in_channels, out_channels=out_channels, base_filters=base_filters)
    
    init_weights(gen)
    init_weights(disc)
    
    gen = gen.to(device)
    disc = disc.to(device)
    return gen, disc

def build_cyclegan_models(
    in_channels: int = 3,
    out_channels: int = 1,
    device: torch.device = torch.device("cpu")
) -> Tuple[ResNetGenerator, CycleGANDiscriminator]:
    """Instantiate and initialize CycleGAN Generator and Discriminator."""
    gen = ResNetGenerator(in_channels=in_channels, out_channels=out_channels)
    disc = CycleGANDiscriminator(in_channels=out_channels)
    
    init_weights(gen)
    init_weights(disc)
    
    gen = gen.to(device)
    disc = disc.to(device)
    return gen, disc
