# -*- coding: utf-8 -*-
"""
Author: Mason Lovejoy-Johnson
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import logging
from tqdm import tqdm
import json
from datetime import datetime

from models import GeneratorPlanner, DiscriminatorPlanner, EMAModel


@dataclass
class TrainingConfig:
    feature_size: int = 42
    hidden_dim: int = 256
    filter_num: int = 4
    p: int = 3
    q: int = 1 

    batch_size: int = 32
    num_iterations: int = 40000
    d_steps_per_g_step: int = 5
    
    # TTUR
    g_lr: float = 1e-4
    d_lr: float = 4e-4
    betas: Tuple[float, float] = (0.0, 0.9)
    
    gradient_clip: float = 1.0
    gradient_penalty_weight: float = 10.0
    
    label_smoothing: bool = True
    real_label_range: Tuple[float, float] = (0.7, 1.2)
    fake_label_range: Tuple[float, float] = (0.0, 0.3)
    
    instance_noise_std: float = 0.1
    noise_decay: float = 0.9999

    use_ema: bool = True
    ema_decay: float = 0.999
    
    use_feature_matching: bool = True
    feature_matching_weight: float = 0.1
    use_minibatch_stddev: bool = True
    
    log_interval: int = 50
    checkpoint_interval: int = 1000
    
    data_path: str = 'data/nvda_data_tensor'
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'


class MetricsTracker:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.reset()
        
    def reset(self):
        self.g_losses = []
        self.d_losses_real = []
        self.d_losses_fake = []
        self.d_losses_total = []
        self.g_adv_losses = []
        self.g_recon_losses = []
        
    def update(self, g_loss: float, d_loss_real: float, d_loss_fake: float, 
               g_adv: float = 0.0, g_recon: float = 0.0):
        self.g_losses.append(g_loss)
        self.d_losses_real.append(d_loss_real)
        self.d_losses_fake.append(d_loss_fake)
        self.d_losses_total.append(d_loss_real + d_loss_fake)
        self.g_adv_losses.append(g_adv)
        self.g_recon_losses.append(g_recon)
        
    def get_recent_avg(self) -> Optional[Dict[str, float]]:
        if len(self.g_losses) < self.window_size:
            return None
            
        return {
            'g_loss_avg': np.mean(self.g_losses[-self.window_size:]),
            'd_loss_avg': np.mean(self.d_losses_total[-self.window_size:]),
            'd_loss_real_avg': np.mean(self.d_losses_real[-self.window_size:]),
            'd_loss_fake_avg': np.mean(self.d_losses_fake[-self.window_size:]),
            'g_adv_avg': np.mean(self.g_adv_losses[-self.window_size:]),
            'g_recon_avg': np.mean(self.g_recon_losses[-self.window_size:])
        }
    
    def get_all_metrics(self) -> Dict[str, List[float]]:
        return {
            'g_losses': self.g_losses,
            'd_losses_real': self.d_losses_real,
            'd_losses_fake': self.d_losses_fake,
            'd_losses_total': self.d_losses_total
        }


class GANTrainer:
    def __init__(self, config: TrainingConfig, G, D, device='cpu'):
        self.config = config
        self.G = G.to(device)
        self.D = D.to(device)
        self.device = device
        
        if config.use_ema:
            self.G_ema = EMAModel(G, decay=config.ema_decay)
        else:
            self.G_ema = None
            
        self.G_optimizer = torch.optim.Adam(
            G.parameters(),
            lr=config.g_lr,
            betas=config.betas
        )
        self.D_optimizer = torch.optim.Adam(
            D.parameters(),
            lr=config.d_lr,
            betas=config.betas
        )
        
        self.current_noise_std = config.instance_noise_std
        
    def compute_gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        
        d_interpolates = self.D(interpolates)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
        
    def add_instance_noise(self, x):
        """Add instance noise to inputs."""
        if self.current_noise_std > 0:
            noise = torch.randn_like(x) * self.current_noise_std
            return x + noise
        return x
        
    def get_labels(self, batch_size, is_real=True):
        if not self.config.label_smoothing:
            labels = torch.ones(batch_size, 1) if is_real else torch.zeros(batch_size, 1)
            return labels.to(self.device)
            
        if is_real:
            labels = torch.FloatTensor(batch_size, 1).uniform_(*self.config.real_label_range)
        else:
            labels = torch.FloatTensor(batch_size, 1).uniform_(*self.config.fake_label_range)
        return labels.to(self.device)
        
    def D_trainstep(self, x_fake, x_real):
        self.D.train()
        self.D_optimizer.zero_grad()
        
        batch_size = x_real.size(0)
        
        x_real_noisy = self.add_instance_noise(x_real)
        x_fake_noisy = self.add_instance_noise(x_fake)
        x_real_noisy = x_real_noisy.unsqueeze(1)   # (B,1,T,F)
        x_fake_noisy = x_fake_noisy.unsqueeze(1)
        
        d_real = self.D(x_real_noisy)
        real_labels = self.get_labels(batch_size, is_real=True)
        loss_real = nn.functional.binary_cross_entropy_with_logits(d_real, real_labels)
        
        d_fake = self.D(x_fake_noisy.detach())
        fake_labels = self.get_labels(batch_size, is_real=False)
        loss_fake = nn.functional.binary_cross_entropy_with_logits(d_fake, fake_labels)
        
        x_real = x_real.unsqueeze(1)
        x_fake = x_fake.detach().unsqueeze(1)
        gp = self.compute_gradient_penalty(x_real, x_fake.detach())

        d_loss = loss_real + loss_fake + self.config.gradient_penalty_weight * gp
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.config.gradient_clip)
        
        self.D_optimizer.step()
        self.current_noise_std *= self.config.noise_decay
        
        return loss_real.item(), loss_fake.item(), gp.item()
        
    def G_trainstep(self, x_real):
        self.G.train()
        self.G_optimizer.zero_grad()
        
        batch_size = x_real.size(0)
        x_past = x_real[:, :self.config.p]
        x_fake = self.G(x_past)
        x_fake_full = torch.cat([x_past, x_fake], dim=1)

        x_fake_full = x_fake_full.unsqueeze(1)  # (B,1,T,F)
        x_real = x_real.unsqueeze(1)
        
        if self.config.use_feature_matching:
            d_fake, fake_features = self.D(x_fake_full, return_features=True)
            with torch.no_grad():
                _, real_features = self.D(x_real, return_features=True)
            feature_loss = torch.mean((fake_features - real_features) ** 2)
        else:
            d_fake = self.D(x_fake_full)
            feature_loss = 0.0
        
        real_labels = torch.ones(batch_size, 1).to(self.device)
        g_adv_loss = nn.functional.binary_cross_entropy_with_logits(d_fake, real_labels)
        g_recon_loss = torch.mean((x_fake_full - x_real) ** 2)
        total_g_loss = g_adv_loss + g_recon_loss

        if self.config.use_feature_matching and isinstance(feature_loss, torch.Tensor):
            total_g_loss += self.config.feature_matching_weight * feature_loss
        
        total_g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.config.gradient_clip)
        
        self.G_optimizer.step()
        
        if self.G_ema is not None:
            self.G_ema.update()
        
        return total_g_loss.item(), g_adv_loss.item(), g_recon_loss.item()


class CGANTrainingSystem:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.setup_directories()
        self.setup_logging()
        self.load_data()
        self.build_models()
        
        self.metrics = MetricsTracker(window_size=config.log_interval)
        self.best_g_loss = float('inf')
        self.iteration = 0
        
    def setup_directories(self):
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(self.config.log_dir) / f'training_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Configuration: {self.config}")
        
    def load_data(self):
        try:
            self.data = torch.load(self.config.data_path).float()
            self.logger.info(f"Loaded data with shape: {self.data.shape}")
            self.data = torch.clamp(self.data, -3, 3) / 3.0
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
            
    def build_models(self):
        self.G = GeneratorPlanner(
            feature_size=self.config.feature_size,
            hidden_dim=self.config.hidden_dim,
            window_size=self.config.p,
            target_size=self.config.q
        )
        
        self.D = DiscriminatorPlanner(
            features=self.config.feature_size,
            filter_num=self.config.filter_num,
            use_minibatch_stddev=self.config.use_minibatch_stddev
        )
        
        self.trainer = GANTrainer(self.config, self.G, self.D, device=self.device)
        
        g_params = sum(p.numel() for p in self.G.parameters())
        d_params = sum(p.numel() for p in self.D.parameters())
        self.logger.info(f"Generator parameters: {g_params:,}")
        self.logger.info(f"Discriminator parameters: {d_params:,}")
        
    def sample_batch(self) -> torch.Tensor:
        indices = np.random.choice(
            self.data.shape[0],
            size=self.config.batch_size,
            replace=True
        )
        return self.data[indices].to(self.device)
        
    def train_step(self):
        d_losses_real, d_losses_fake = [], []
        for _ in range(self.config.d_steps_per_g_step):
            x_real = self.sample_batch()
            x_past = x_real[:, :self.config.p]
            
            with torch.no_grad():
                x_fake = self.G(x_past)
                x_fake_full = torch.cat([x_past, x_fake], dim=1)
            
            d_loss_real, d_loss_fake, gp = self.trainer.D_trainstep(x_fake_full, x_real)
            d_losses_real.append(d_loss_real)
            d_losses_fake.append(d_loss_fake)
        
        x_real = self.sample_batch()
        g_loss, g_adv, g_recon = self.trainer.G_trainstep(x_real)
        
        return {
            'g_loss': g_loss,
            'd_loss_real': np.mean(d_losses_real),
            'd_loss_fake': np.mean(d_losses_fake),
            'g_adv': g_adv,
            'g_recon': g_recon
        }
        
    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'iteration': self.iteration,
            'G_state_dict': self.G.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'G_optimizer': self.trainer.G_optimizer.state_dict(),
            'D_optimizer': self.trainer.D_optimizer.state_dict(),
            'metrics': self.metrics.get_all_metrics(),
            'config': self.config,
            'best_g_loss': self.best_g_loss
        }
        
        if is_best:
            path = Path(self.config.checkpoint_dir) / 'best_model.pt'
            torch.save(checkpoint, path)
            self.logger.info(f"Saved best model at iteration {self.iteration}")
        else:
            path = Path(self.config.checkpoint_dir) / f'checkpoint_{self.iteration}.pt'
            torch.save(checkpoint, path)
            
    def plot_training_curves(self):
        metrics = self.metrics.get_all_metrics()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(metrics['g_losses'], alpha=0.6)
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)

        axes[0, 1].plot(metrics['d_losses_total'], alpha=0.6)
        axes[0, 1].set_title('Discriminator Loss')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(metrics['d_losses_real'], alpha=0.6, label='Real')
        axes[1, 0].plot(metrics['d_losses_fake'], alpha=0.6, label='Fake')
        axes[1, 0].set_title('D Loss: Real vs Fake')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        window = 100
        if len(metrics['g_losses']) >= window:
            g_smooth = np.convolve(metrics['g_losses'], np.ones(window)/window, mode='valid')
            d_smooth = np.convolve(metrics['d_losses_total'], np.ones(window)/window, mode='valid')
            axes[1, 1].plot(g_smooth, label='G Loss (smoothed)')
            axes[1, 1].plot(d_smooth, label='D Loss (smoothed)')
            axes[1, 1].set_title('Smoothed Losses')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig_path = Path(self.config.log_dir) / f'training_curves_{timestamp}.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved training curves: {fig_path}")
        
    def train(self):
        self.logger.info("=" * 60)
        self.logger.info("Starting Training")
        self.logger.info("=" * 60)
        
        try:
            pbar = tqdm(range(self.config.num_iterations), desc="Training")
            
            for self.iteration in pbar:
                losses = self.train_step()
                self.metrics.update(
                    losses['g_loss'],
                    losses['d_loss_real'],
                    losses['d_loss_fake'],
                    losses['g_adv'],
                    losses['g_recon']
                )
                
                pbar.set_postfix({
                    'G': f"{losses['g_loss']:.4f}",
                    'D': f"{losses['d_loss_real'] + losses['d_loss_fake']:.4f}"
                })
                
                if (self.iteration + 1) % self.config.log_interval == 0:
                    metrics_avg = self.metrics.get_recent_avg()
                    if metrics_avg:
                        self.logger.info(
                            f"Iter {self.iteration+1:6d} | "
                            f"G: {metrics_avg['g_loss_avg']:.4f} | "
                            f"D: {metrics_avg['d_loss_avg']:.4f} | "
                            f"G_adv: {metrics_avg['g_adv_avg']:.4f} | "
                            f"G_recon: {metrics_avg['g_recon_avg']:.4f}"
                        )
                        
                        is_best = metrics_avg['g_loss_avg'] < self.best_g_loss
                        if is_best:
                            self.best_g_loss = metrics_avg['g_loss_avg']
                            self.save_checkpoint(is_best=True)
                
                if (self.iteration + 1) % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(is_best=False)
                    
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            self.save_checkpoint(is_best=False)
            self.plot_training_curves()
            self.logger.info("Training completed")
            
    def generate_summary(self):
        metrics = self.metrics.get_all_metrics()
        
        summary = {
            'total_iterations': self.iteration + 1,
            'best_g_loss': self.best_g_loss,
            'final_g_loss': metrics['g_losses'][-1] if metrics['g_losses'] else None,
            'final_d_loss': metrics['d_losses_total'][-1] if metrics['d_losses_total'] else None,
            'min_g_loss': min(metrics['g_losses']) if metrics['g_losses'] else None,
            'config': vars(self.config)
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = Path(self.config.log_dir) / f'summary_{timestamp}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Saved summary: {summary_path}")
        
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total Iterations: {summary['total_iterations']}")
        print(f"Best Generator Loss: {summary['best_g_loss']:.6f}")
        print(f"Final Generator Loss: {summary['final_g_loss']:.6f}")
        print(f"Final Discriminator Loss: {summary['final_d_loss']:.6f}")
        print(f"Minimum Generator Loss: {summary['min_g_loss']:.6f}")
        print("=" * 60)


def main():
    config = TrainingConfig(
        batch_size=32,
        num_iterations=40000,
        d_steps_per_g_step=5,
        log_interval=50,
        checkpoint_interval=1000,
        label_smoothing=True,
        use_ema=True,
        use_feature_matching=True,
        use_minibatch_stddev=True
    )
    
    trainer = CGANTrainingSystem(config)
    trainer.train()
    trainer.generate_summary()


if __name__ == "__main__":
    main()