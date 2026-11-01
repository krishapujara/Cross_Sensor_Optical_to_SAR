"""
Loss functions for conditional generative adversarial optical-to-SAR translation.
"""
import torch
import torch.nn as nn
from typing import Tuple, Dict

class GANLoss(nn.Module):
    """Adversarial loss module supporting BCEWithLogits and LSGAN (MSE)."""
    def __init__(self, use_lsgan: bool = False, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.use_lsgan = use_lsgan
        if use_lsgan:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            
    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Create target label tensor matching prediction spatial dimensions."""
        if target_is_real:
            return torch.ones_like(prediction, device=self.device)
        else:
            return torch.zeros_like(prediction, device=self.device)
            
    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss_fn(prediction, target_tensor)

class Pix2PixLossComputer:
    """Computes combined generator and discriminator loss objectives."""
    def __init__(self, lambda_l1: float = 100.0, use_lsgan: bool = False, device: torch.device = torch.device("cpu")):
        self.lambda_l1 = lambda_l1
        self.criterion_gan = GANLoss(use_lsgan=use_lsgan, device=device)
        self.criterion_l1 = nn.L1Loss()
        
    def compute_discriminator_loss(
        self,
        discriminator: nn.Module,
        optical: torch.Tensor,
        real_sar: torch.Tensor,
        fake_sar: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        D loss = 0.5 * (D_real_loss + D_fake_loss)
        """
        # Real SAR target
        pred_real = discriminator(optical, real_sar)
        loss_d_real = self.criterion_gan(pred_real, target_is_real=True)
        
        # Fake SAR output (detached from generator graph to prevent backprop into G)
        pred_fake = discriminator(optical, fake_sar.detach())
        loss_d_fake = self.criterion_gan(pred_fake, target_is_real=False)
        
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        logs = {
            "loss_d": loss_d.item(),
            "loss_d_real": loss_d_real.item(),
            "loss_d_fake": loss_d_fake.item()
        }
        return loss_d, logs
        
    def compute_generator_loss(
        self,
        discriminator: nn.Module,
        optical: torch.Tensor,
        real_sar: torch.Tensor,
        fake_sar: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        G loss = GAN_loss(D(optical, fake_sar), 1) + lambda_l1 * L1(real_sar, fake_sar)
        """
        # Fool the discriminator
        pred_fake = discriminator(optical, fake_sar)
        loss_g_gan = self.criterion_gan(pred_fake, target_is_real=True)
        
        # Reconstruction L1 loss
        loss_g_l1 = self.criterion_l1(fake_sar, real_sar) * self.lambda_l1
        
        loss_g = loss_g_gan + loss_g_l1
        logs = {
            "loss_g": loss_g.item(),
            "loss_g_gan": loss_g_gan.item(),
            "loss_g_l1": loss_g_l1.item()
        }
        return loss_g, logs
