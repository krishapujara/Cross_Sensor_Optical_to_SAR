"""
U-Net Generator architecture for Pix2Pix optical-to-SAR translation.
Includes skip connections, downsampling encoders, bottleneck, and upsampling decoders with Tanh output.
"""
import torch
import torch.nn as nn
from typing import List

class UNetDownBlock(nn.Module):
    """Downsampling block: Conv2d -> [Norm] -> LeakyReLU."""
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class UNetUpBlock(nn.Module):
    """Upsampling block: ConvTranspose2d -> Norm -> [Dropout] -> ReLU -> Skip concat."""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return torch.cat((x, skip_input), dim=1)

class UNetGenerator(nn.Module):
    """
    U-Net 256 Generator with skip connections.
    Takes optical satellite image (e.g. 3 channels) and outputs synthetic SAR image (1 or 2 channels).
    Output activation is Tanh producing values in range [-1.0, 1.0].
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_filters: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder (Downsampling)
        # 256 -> 128
        self.down1 = UNetDownBlock(in_channels, base_filters, normalize=False)
        # 128 -> 64
        self.down2 = UNetDownBlock(base_filters, base_filters * 2)
        # 64 -> 32
        self.down3 = UNetDownBlock(base_filters * 2, base_filters * 4)
        # 32 -> 16
        self.down4 = UNetDownBlock(base_filters * 4, base_filters * 8)
        # 16 -> 8
        self.down5 = UNetDownBlock(base_filters * 8, base_filters * 8)
        # 8 -> 4
        self.down6 = UNetDownBlock(base_filters * 8, base_filters * 8)
        # 4 -> 2
        self.down7 = UNetDownBlock(base_filters * 8, base_filters * 8)
        # 2 -> 1 (Bottleneck)
        self.down8 = UNetDownBlock(base_filters * 8, base_filters * 8, normalize=False)
        
        # Decoder (Upsampling with Skip connections)
        # 1 -> 2 (skip: 512 + 512 = 1024)
        self.up1 = UNetUpBlock(base_filters * 8, base_filters * 8, dropout=0.5)
        # 2 -> 4
        self.up2 = UNetUpBlock(base_filters * 16, base_filters * 8, dropout=0.5)
        # 4 -> 8
        self.up3 = UNetUpBlock(base_filters * 16, base_filters * 8, dropout=0.5)
        # 8 -> 16
        self.up4 = UNetUpBlock(base_filters * 16, base_filters * 8, dropout=0.0)
        # 16 -> 32
        self.up5 = UNetUpBlock(base_filters * 16, base_filters * 4, dropout=0.0)
        # 32 -> 64
        self.up6 = UNetUpBlock(base_filters * 8, base_filters * 2, dropout=0.0)
        # 64 -> 128
        self.up7 = UNetUpBlock(base_filters * 4, base_filters, dropout=0.0)
        
        # Final layer: 128 -> 256 (no skip concatenation here, directly to output channels)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Maps to [-1.0, 1.0]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass with skip tensors
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # Decoder forward pass with concatenated skips
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final(u7)
