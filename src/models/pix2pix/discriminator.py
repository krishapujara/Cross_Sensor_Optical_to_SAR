"""
PatchGAN Discriminator for Pix2Pix optical-to-SAR translation.
Classifies N x N overlapping patches as real or synthetic radar backscatter.
"""
import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator.
    Takes concatenated [Optical_condition, SAR_target] of shape (B, in_channels + out_channels, H, W).
    Outputs raw logits tensor of shape (B, 1, H/8, W/8).
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_filters: int = 64):
        super().__init__()
        total_input_channels = in_channels + out_channels
        
        # Layer 1: no batch norm
        self.layer1 = nn.Sequential(
            nn.Conv2d(total_input_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Layer 4 (stride 1)
        self.layer4 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_filters * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Final classification layer (stride 1)
        self.final = nn.Conv2d(base_filters * 8, 1, kernel_size=4, stride=1, padding=1)
        
    def forward(self, optical_input: torch.Tensor, sar_target: torch.Tensor) -> torch.Tensor:
        """Concatenate condition and target along channel dimension."""
        combined = torch.cat([optical_input, sar_target], dim=1)
        x = self.layer1(combined)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.final(x)
