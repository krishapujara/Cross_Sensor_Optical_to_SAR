"""
Core PyTorch training loop for Pix2Pix optical-to-SAR translation.
Includes real-time loss tracking, validation loop, callbacks, and checkpoint serialization.
"""
import time
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Callable
from torch.utils.data import DataLoader
from .losses import Pix2PixLossComputer
from .checkpoints import save_checkpoint
from ..evaluation.metrics import evaluate_image_pair
from ..config.settings import CHECKPOINTS_DIR
from ..utils.logging_config import logger

class Pix2PixTrainer:
    """Trainer executing conditional adversarial training for cross-sensor translation."""
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        experiment_id: str = "pix2pix_run",
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        lambda_l1: float = 100.0,
        checkpoint_dir = CHECKPOINTS_DIR,
        device: torch.device = torch.device("cpu")
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_id = experiment_id
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        base_filters = getattr(generator, 'down1', None)
        base_f_val = generator.down1.block[0].out_channels if hasattr(generator, 'down1') else 64
        self.hyperparameters = {
            "learning_rate": lr,
            "beta1": beta1,
            "beta2": beta2,
            "lambda_l1": lambda_l1,
            "base_filters": base_f_val,
            "device": str(device),
            "model": "Pix2Pix (UNet256 + PatchGAN)"
        }
        
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        self.loss_computer = Pix2PixLossComputer(lambda_l1=lambda_l1, device=device)
        
        # Training tracking history
        self.history: List[Dict[str, Any]] = []
        self.is_interrupted = False
        
    def stop_training(self):
        """Signal trainer to halt execution at next iteration."""
        self.is_interrupted = True
        logger.info("Training interruption requested by user/process.")
        
    def train(
        self,
        epochs: int = 5,
        checkpoint_freq: int = 1,
        epoch_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        step_callback: Optional[Callable[[int, int, float, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute training across specified epochs.
        Returns dictionary containing full training history, best checkpoint, and execution duration.
        """
        start_time = time.time()
        best_val_psnr = -1.0
        best_checkpoint_path = None
        self.is_interrupted = False
        
        logger.info(f"Starting Pix2Pix training for {epochs} epochs on device: {self.device}")
        
        for epoch in range(1, epochs + 1):
            if self.is_interrupted:
                logger.info(f"Training stopped early at epoch {epoch - 1}")
                break
                
            self.generator.train()
            self.discriminator.train()
            
            epoch_loss_g = 0.0
            epoch_loss_d = 0.0
            batch_count = 0
            
            for batch_idx, (opt_batch, sar_batch, _) in enumerate(self.train_loader):
                if self.is_interrupted:
                    break
                    
                opt_batch = opt_batch.to(self.device)
                sar_batch = sar_batch.to(self.device)
                
                # ---------------------
                # 1. Forward Generator
                # ---------------------
                fake_sar = self.generator(opt_batch)
                
                # ---------------------
                # 2. Update Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()
                loss_d, d_logs = self.loss_computer.compute_discriminator_loss(
                    self.discriminator, opt_batch, sar_batch, fake_sar
                )
                loss_d.backward()
                self.optimizer_D.step()
                
                # ---------------------
                # 3. Update Generator
                # ---------------------
                self.optimizer_G.zero_grad()
                loss_g, g_logs = self.loss_computer.compute_generator_loss(
                    self.discriminator, opt_batch, sar_batch, fake_sar
                )
                loss_g.backward()
                self.optimizer_G.step()
                
                epoch_loss_g += loss_g.item()
                epoch_loss_d += loss_d.item()
                batch_count += 1
                
                if step_callback:
                    step_callback(batch_idx + 1, len(self.train_loader), loss_g.item(), loss_d.item())
                    
            avg_loss_g = epoch_loss_g / max(1, batch_count)
            avg_loss_d = epoch_loss_d / max(1, batch_count)
            
            # ---------------------
            # 4. Validation Loop
            # ---------------------
            val_metrics = self.validate()
            is_best = val_metrics.get("psnr", 0.0) > best_val_psnr
            if is_best:
                best_val_psnr = val_metrics.get("psnr", 0.0)
                
            epoch_data = {
                "epoch": epoch,
                "loss_g": round(avg_loss_g, 4),
                "loss_d": round(avg_loss_d, 4),
                "val_psnr": val_metrics.get("psnr", 0.0),
                "val_ssim": val_metrics.get("ssim", 0.0),
                "val_mae": val_metrics.get("mae", 0.0),
                "val_mse": val_metrics.get("mse", 0.0),
                "is_best": is_best
            }
            self.history.append(epoch_data)
            logger.info(
                f"[Epoch {epoch}/{epochs}] Loss_G: {avg_loss_g:.4f} | Loss_D: {avg_loss_d:.4f} | "
                f"Val PSNR: {val_metrics.get('psnr'):.2f} dB | Val SSIM: {val_metrics.get('ssim'):.4f}"
            )
            
            # Save Checkpoint
            if epoch % checkpoint_freq == 0 or epoch == epochs or is_best:
                ckpt_path = save_checkpoint(
                    checkpoint_dir=self.checkpoint_dir,
                    experiment_id=self.experiment_id,
                    epoch=epoch,
                    generator=self.generator,
                    discriminator=self.discriminator,
                    optimizer_G=self.optimizer_G,
                    optimizer_D=self.optimizer_D,
                    metrics=val_metrics,
                    hyperparameters=self.hyperparameters,
                    is_best=is_best
                )
                if is_best:
                    best_checkpoint_path = ckpt_path
                    
            if epoch_callback:
                epoch_callback(epoch_data)
                
        total_duration = round(time.time() - start_time, 2)
        return {
            "experiment_id": self.experiment_id,
            "total_duration_sec": total_duration,
            "epochs_completed": len(self.history),
            "history": self.history,
            "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path else None,
            "best_val_psnr": best_val_psnr,
            "final_metrics": self.history[-1] if self.history else {}
        }
        
    def validate(self) -> Dict[str, float]:
        """Compute real evaluation metrics across held-out validation samples."""
        if not self.val_loader:
            return {"psnr": 0.0, "ssim": 0.0, "mae": 0.0, "mse": 0.0}
            
        self.generator.eval()
        psnr_list, ssim_list, mae_list, mse_list = [], [], [], []
        
        with torch.no_grad():
            for opt_batch, sar_batch, _ in self.val_loader:
                opt_batch = opt_batch.to(self.device)
                sar_batch = sar_batch.to(self.device)
                fake_sar = self.generator(opt_batch)
                
                # Evaluate each image in batch
                for i in range(opt_batch.size(0)):
                    m = evaluate_image_pair(sar_batch[i], fake_sar[i])
                    psnr_list.append(m["psnr"])
                    ssim_list.append(m["ssim"])
                    mae_list.append(m["mae"])
                    mse_list.append(m["mse"])
                    
        return {
            "psnr": round(float(torch.tensor(psnr_list).mean().item()), 3) if psnr_list else 0.0,
            "ssim": round(float(torch.tensor(ssim_list).mean().item()), 4) if ssim_list else 0.0,
            "mae": round(float(torch.tensor(mae_list).mean().item()), 4) if mae_list else 0.0,
            "mse": round(float(torch.tensor(mse_list).mean().item()), 4) if mse_list else 0.0,
        }
