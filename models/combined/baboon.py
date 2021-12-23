import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from ..adv_gan_lightning.adv_gan import AdvGAN
from ..ape_gan_lightning.ape_gan import ApeGan

from torchmetrics.functional import accuracy
import wandb

class Baboon(LightningModule):
    def __init__(
            self,
            adv_gan: AdvGAN,
            ape_gan: ApeGan,
            lr: float = 0.001,
            num_batches_to_log = 1,
            num_samples_to_log = 16,
            **kwargs
    ):
        super().__init__()
        
        self.adv_gan = adv_gan
        self.ape_gan = ape_gan
        self.target_model = self.adv_gan.target_model

        self.lr = lr

        self.start_epoch_combined = 10

        self.num_batches_to_log = num_batches_to_log
        self.num_samples_to_log = num_samples_to_log

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch

        if optimizer_idx == 0 or optimizer_idx == 1:
            if self.current_epoch == self.start_epoch_combined:
                imgs = self.ape_gan(imgs)

            return self.adv_gan.training_step((imgs, labels), batch_idx, optimizer_idx)
        elif optimizer_idx == 2 or optimizer_idx == 3:
            if self.current_epoch == self.start_epoch_combined:
                imgs = self.adv_gan(imgs)

            return self.ape_gan.training_step((imgs, labels), batch_idx, optimizer_idx - 2)
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch

        adv_imgs = self.adv_gan(imgs)
        res_imgs = self.ape_gan(adv_imgs)

        return imgs, labels, adv_imgs, res_imgs

    def validation_epoch_end(self, outputs):
        imgs_batches, labels_batches, adv_imgs_batches, res_imgs_batches = [torch.stack([output[i] for output in outputs])[:self.num_batches_to_log, :self.num_samples_to_log] for i in range(len(outputs[0]))]

        wandb.log({
            "pred_imgs": [
                wandb.Image(
                    img,
                    caption=f'Pred: {label}, Label: {label}'
                ) for imgs, labels in zip(imgs_batches, labels_batches) for img, label in zip(imgs, labels)
            ] if self.current_epoch == 0 else None,
            "pred_adv_imgs": [
                wandb.Image(
                    adv_img,
                    caption=f'Pred: {label}, Label: {label}'
                ) for adv_imgs, labels in zip(adv_imgs_batches, labels_batches) for adv_img, label in zip(adv_imgs, labels)
            ],
            "pred_res_imgs": [
                wandb.Image(
                    res_img,
                    caption=f'Label: {label}'
                ) for res_imgs, labels in zip(res_imgs_batches, labels_batches) for res_img, label in zip(res_imgs, labels)
            ]
        })

    def configure_optimizers(self):
        adv_gan_opt_g, adv_gan_opt_d = self.adv_gan.configure_optimizers()[0]
        ape_gan_opt_g, ape_gan_opt_d = self.ape_gan.configure_optimizers()[0]
        
        return adv_gan_opt_g, adv_gan_opt_d, ape_gan_opt_g, ape_gan_opt_d