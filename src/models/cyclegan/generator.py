"""
ResNet-based Generator architecture for unpaired CycleGAN satellite translation.
"""
import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    """Residual Block with ReflectionPad2d and InstanceNorm2d."""
    def __init__(self, dim: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            nn.InstanceNorm2d(dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)

class ResNetGenerator(nn.Module):
    """ResNet Generator with 6 or 9 residual blocks for unpaired CycleGAN."""
    def __init__(self, in_channels: int = 3, out_channels: int = 1, num_blocks: int = 6, base_filters: int = 64):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_filters, kernel_size=7, bias=False),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        n_down = 2
        for i in range(n_down):
            mult = 2 ** i
            model += [
                nn.Conv2d(base_filters * mult, base_filters * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(base_filters * mult * 2),
                nn.ReLU(inplace=True)
            ]
            
        # Residual blocks
        mult = 2 ** n_down
        for _ in range(num_blocks):
            model += [ResNetBlock(base_filters * mult)]
            
        # Upsampling
        for i in range(n_down):
            mult = 2 ** (n_down - i)
            model += [
                nn.ConvTranspose2d(base_filters * mult, int(base_filters * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(int(base_filters * mult / 2)),
                nn.ReLU(inplace=True)
            ]
            
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_filters, out_channels, kernel_size=7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
