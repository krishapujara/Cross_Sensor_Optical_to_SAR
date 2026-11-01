"""
PatchGAN Discriminator for CycleGAN (operates on unconditioned single-domain inputs).
"""
import torch
import torch.nn as nn

class CycleGANDiscriminator(nn.Module):
    """PatchGAN Discriminator for unpaired CycleGAN evaluation."""
    def __init__(self, in_channels: int = 1, base_filters: int = 64):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        mult = 1
        for i in range(1, 3):
            prev_mult = mult
            mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(base_filters * prev_mult, base_filters * mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(base_filters * mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            
        layers += [
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(base_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters * 8, 1, kernel_size=4, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
