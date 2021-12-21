import pytorch_lightning as pl
import torch
from torch import nn

from .models import MnistCNN, CifarCNN, Generator, Discriminator
from ..adv_gan_lightning.target_model import TargetModel

import wandb

class ApeGan(pl.LightningModule):
    def __init__(
            self, 
            in_ch=1, 
            xi1=0.7, 
            xi2=0.3, 
            lr=2e-4, 
            checkpoint_path: str ='last.ckpt', 
            attack=None,
            target_checkpoint_path=None,
            num_batches_to_log = 1,
            num_samples_to_log = 16,
        ):
        super().__init__()
        
        self.xi1 = xi1
        self.xi2 = xi2
        self.lr = lr
        self.checkpoint_path = checkpoint_path
        
        self.generator = Generator(in_ch)
        self.discriminator = Discriminator(in_ch)
        
        self.automatic_optimization = False

        self.attack = attack

        self.loss_bce = nn.BCEWithLogitsLoss()
        self.loss_mse = nn.MSELoss()

        self.num_batches_to_log = num_batches_to_log
        self.num_samples_to_log = num_samples_to_log

        if target_checkpoint_path is not None:
            self.target_model = TargetModel.load_from_checkpoint(checkpoint_path=target_checkpoint_path)

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        X, X_adv = batch

        if self.attack is not None:
            y = X_adv.clone()
            X_adv = self.attack(X)

        t_real = torch.ones(X.shape[0], device=self.device)
        t_fake = torch.zeros(X.shape[0], device=self.device)

        # Train discriminator
        opt_d, opt_g = self.optimizers()
        y_real = self.discriminator(X).squeeze()
        X_fake = self.generator(X_adv)
        y_fake = self.discriminator(X_fake).squeeze()

        loss_discriminator = self.loss_bce(y_real, t_real) + self.loss_bce(y_fake, t_fake)
        
        opt_d.zero_grad()
        # retain graph probably wrong
        self.manual_backward(loss_discriminator, retain_graph=True)
        opt_d.step()

        # Train generator
        for _ in range(2):
            X_fake = self.generator(X_adv)
            y_fake = self.discriminator(X_fake).squeeze()

            loss_generator = self.xi1 * self.loss_mse(X_fake, X) + self.xi2 * self.loss_bce(y_fake, t_real)
            
            opt_g.zero_grad()
            self.manual_backward(loss_generator, retain_graph=True)
            opt_g.step()

        losses = {
            "train_loss_discriminiator": loss_discriminator,
            "train_loss_generator": loss_generator
        }

        self.log_dict(
            losses,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )

    def validation_step(self, batch, batch_idx):
        X, X_adv = batch

        if self.attack is not None:
            y = X_adv.clone()
            X_adv = self.attack(X)

        t_real = torch.ones(X.shape[0], device=self.device)
        t_fake = torch.zeros(X.shape[0], device=self.device)

        y_real = self.discriminator(X).squeeze()
        X_fake = self.generator(X_adv)
        y_fake = self.discriminator(X_fake).squeeze()

        loss_discriminator = self.loss_bce(y_real, t_real) + self.loss_bce(y_fake, t_fake)
        loss_generator = self.xi1 * self.loss_mse(X_fake, X) + self.xi2 * self.loss_bce(y_fake, t_real)

        losses = {
            "validation_loss_discriminiator": loss_discriminator,
            "validation_loss_generator": loss_generator
        }

        self.log_dict(
            losses,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )

        return X, y, X_adv, X_fake

    def validation_epoch_end(self, outputs):
        imgs_batches, labels_batches, adv_imgs_batches, res_imgs_batches = [torch.stack([output[i] for output in outputs])[:self.num_batches_to_log, :self.num_samples_to_log] for i in range(len(outputs[0]))]
        preds_batches = labels_batches

        wandb.log({
            "original_imgs": [
                wandb.Image(
                    img,
                    caption=f'Pred: {pred}, Label: {label}'
                ) for imgs, labels, preds in zip(imgs_batches, labels_batches, preds_batches) for img, pred, label in zip(imgs, labels, preds)
            ],
            "attack_imgs": [
                wandb.Image(
                    adv_img,
                    caption=f'Pred: {pred}, Label: {label}'
                ) for adv_imgs, labels, preds in zip(adv_imgs_batches, labels_batches, preds_batches) for adv_img, pred, label in zip(adv_imgs, labels, preds)
            ],
            "restored_imgs": [
                wandb.Image(
                    res_img,
                    caption=f'Pred: {pred}, Label: {label}'
                ) for res_imgs, labels, preds in zip(res_imgs_batches, labels_batches, preds_batches) for res_img, pred, label in zip(res_imgs, labels, preds)
            ],
        })

    def configure_optimizers(self):
        lr = self.lr
        
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        return opt_d, opt_g