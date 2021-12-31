import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from ..adv_gan.adv_gan import AdvGAN
from ..ape_gan.ape_gan import ApeGan

from torchmetrics.functional import accuracy
import wandb


class Baboon(LightningModule):
    def __init__(
            self,
            adv_gan: AdvGAN,
            ape_gan: ApeGan,
            lr: float = 0.001,
            num_batches_to_log=1,
            num_samples_to_log=16,
            **kwargs
    ):
        super().__init__()

        self.adv_gan = adv_gan
        self.ape_gan = ape_gan
        self.target_model = self.adv_gan.target_model

        self.lr = lr

        self.start_epoch_combined = 5

        self.attack_prob = 0.8
        self.defense_prob = 0.5

        self.num_batches_to_log = num_batches_to_log
        self.num_samples_to_log = num_samples_to_log

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch

        if optimizer_idx == 0 or optimizer_idx == 1:
            if self.current_epoch >= self.start_epoch_combined and torch.rand(1) < self.defense_prob:
                imgs = self.ape_gan(imgs)

            return self.adv_gan.training_step((imgs, labels), batch_idx, optimizer_idx)

        if optimizer_idx == 2 or optimizer_idx == 3:
            if self.current_epoch >= self.start_epoch_combined and torch.rand(1) < self.attack_prob:
                imgs = self.adv_gan(imgs)

            return self.ape_gan.training_step((imgs, labels), batch_idx, optimizer_idx - 2)

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch

        adv_imgs = self.adv_gan(imgs)
        res_imgs = self.ape_gan(adv_imgs)

        y_original_pred = self.target_model(imgs).argmax(1)
        y_adversarial_pred = self.target_model(adv_imgs).argmax(1)
        y_restored_pred = self.target_model(res_imgs).argmax(1)

        accuracy_original = accuracy(y_original_pred, labels)
        accuracy_adversarial = accuracy(y_adversarial_pred, labels)
        accuracy_restored = accuracy(y_restored_pred, labels)

        metrics = {
            "baboon_validation_accuracy_original": accuracy_original,
            "baboon_validation_accuracy_adversarial": accuracy_adversarial,
            "baboon_validation_accuracy_restored": accuracy_restored,
        }

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )

        return imgs, labels, adv_imgs, res_imgs, y_original_pred, y_adversarial_pred, y_restored_pred

    def validation_epoch_end(self, outputs):
        imgs_batches, labels_batches, adv_imgs_batches, res_imgs_batches, y_original_pred_batches, y_adversarial_pred_batches, y_restored_pred_batches = [
            torch.stack([output[i] for output in outputs])[:self.num_batches_to_log, :self.num_samples_to_log] for i in
            range(len(outputs[0]))]

        wandb.log({
            "pred_imgs": [
                wandb.Image(
                    img,
                    caption=f'Pred: {pred}, Label: {label}'
                ) for imgs, labels, preds in zip(imgs_batches, labels_batches, y_original_pred_batches) for
                img, label, pred in zip(imgs, labels, preds)
            ] if self.current_epoch == 0 else None,
            "pred_adv_imgs": [
                wandb.Image(
                    adv_img,
                    caption=f'Pred: {pred}, Label: {label}'
                ) for adv_imgs, labels, preds in zip(adv_imgs_batches, labels_batches, y_adversarial_pred_batches) for
                adv_img, label, pred in zip(adv_imgs, labels, preds)
            ],
            "pred_res_imgs": [
                wandb.Image(
                    res_img,
                    caption=f'Pred: {pred}, Label: {label}'
                ) for res_imgs, labels, preds in zip(res_imgs_batches, labels_batches, y_restored_pred_batches) for
                res_img, label, pred in zip(res_imgs, labels, preds)
            ]
        })

    def configure_optimizers(self):
        adv_gan_opt_g, adv_gan_opt_d = self.adv_gan.configure_optimizers()[0]
        ape_gan_opt_g, ape_gan_opt_d = self.ape_gan.configure_optimizers()[0]

        return adv_gan_opt_g, adv_gan_opt_d, ape_gan_opt_g, ape_gan_opt_d
