import logging
import time
import torch
import pytorch_lightning as pl
import numpy as np

from torch.nn.functional import avg_pool2d
from pytorch_lightning.callbacks import ProgressBarBase
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchbox.utils.gan import ModelWrapper
from torchbox.losses.gan import BaseGANLossFunction

from msg_gan.utils import create_multi_scale_image_grid


class MultiScaleGradientGAN(pl.LightningModule):
    def __init__(self, hparams, G, D, dataset, loss_f: BaseGANLossFunction, sample_dir, sample_model: ModelWrapper):
        super().__init__()

        self.hparams = hparams

        self.G = G
        self.D = D

        self.dataset = dataset
        self.loss_f = loss_f
        self.sample_dir = sample_dir

        self.register_buffer('z_sample', self.draw_latent(hparams.n_sample))

        # self.sample_model = sample_model

    def forward(self, z):
        return self.G(z)

    def configure_optimizers(self):
        beta1, beta2 = self.hparams.beta1, self.hparams.beta2
        G_opt = torch.optim.Adam(self.G.parameters(), lr=self.hparams.g_lr, betas=(beta1, beta2))
        D_opt = torch.optim.Adam(self.D.parameters(), lr=self.hparams.d_lr, betas=(beta1, beta2))

        return D_opt, G_opt

    def draw_latent(self, num_samples):
        return torch.randn(num_samples, self.hparams.latent_size)

    def train_G(self, reals):
        z = self.draw_latent(len(reals[0])).to(self.device)
        fakes = self.G(z)
        return self.loss_f.loss_g(reals, fakes)

    def train_D(self, reals):
        z = self.draw_latent(len(reals[0])).to(self.device)
        fakes = self.G(z)
        return self.loss_f.loss_d(reals, fakes)

    def training_step(self, batch, batch_nb, optimizer_idx):
        images = [batch] + [avg_pool2d(batch, int(np.power(2, i)))
                            for i in range(1, self.D.depth)]
        images = images[::-1]

        if optimizer_idx == 0:
            loss = self.train_D(images)

        if optimizer_idx == 1:
            loss = self.train_G(images)

        log_key = 'g_loss' if optimizer_idx == 1 else 'd_loss'
        log_dict = {log_key: float(loss)}
        return {
            'loss': loss,
            'progress_bar': log_dict,
            'log': log_dict
        }

    def train_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=self.hparams.num_workers)

    def on_train_batch_end(self, batch, batch_idx, dataloader_idx):
        if self.global_step % self.hparams.sample_every == 0 and self.sample_dir:
            with torch.no_grad():
                samples = self(self.z_sample.to(self.device))
                image_grid = create_multi_scale_image_grid(samples)
                save_image(image_grid, self.sample_dir / f'{self.global_step:06d}_fakes.png')

                self.logger.experiment.add_image('samples', image_grid, self.global_step)

        # self.sample_model.update(self.G)


class ProgressBar(ProgressBarBase):
    def __init__(self, log_every=1):
        super().__init__()
        self.enable = True

        self.log_every = log_every

        self.acc_stats = {}
        self.acc_count = 0

        self.latest_log_time = time.time()

    def disable(self):
        self.enable = False

    def get_avg_stats(self):
        return {k: v / self.acc_count for k, v in self.acc_stats.items()}

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module, batch, batch_idx, dataloader_idx):
        super().on_batch_end(trainer, pl_module)

        for k, v in trainer.progress_bar_metrics.items():
            self.acc_stats[k] = v + self.acc_stats.get(k, 0)
        self.acc_count += 1

        if trainer.batch_idx % self.log_every == 0:
            time_diff = time.time() - self.latest_log_time
            self.latest_log_time = time.time()

            stats = self.get_avg_stats()

            logging.info(f'[EPOCH {trainer.current_epoch + 1:03d} / {trainer.max_epochs:03d}] '
                         f'[{trainer.batch_idx:05d} / {trainer.num_training_batches:05d}] ' +
                         f'd_loss: {stats["d_loss"]:.5f}, g_loss: {stats["g_loss"]:.5f}, '
                         f'batches/s: {self.acc_count / time_diff:02.2f}')

            self.acc_count = 0
            self.acc_stats = {}
