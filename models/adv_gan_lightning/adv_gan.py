from .discriminator import Discriminator
from .generator import Generator
from .target_model import TargetModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

import wandb

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN(LightningModule):
    def __init__(
            self,
            model_num_labels,
            image_nc,
            box_min,
            box_max,
            lr: float = 0.001,
            b1: float = 0.5,
            b2: float = 0.999,
            checkpoint_path: str = "target.ckpt",
            num_batches_to_log = 1,
            num_samples_to_log = 16,
            **kwargs
    ):
        super().__init__()
        self.b2 = b2
        self.b1 = b1
        self.lr = lr
        self.save_hyperparameters()

        output_nc = image_nc
        self.model_num_labels = model_num_labels
        self.gen_input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max

        # networks
        self.generator = Generator(self.gen_input_nc, image_nc)
        self.discriminator = Discriminator(image_nc)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.model = TargetModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

        self.C = 0.1
        # To scale the importance of losses
        self.gen_lambda = 1
        self.adv_lambda = 10
        self.pert_lambda = 1

        self.num_batches_to_log = num_batches_to_log
        self.num_samples_to_log = num_samples_to_log

    def forward(self, z):
        perturbations, adv_imgs = self.generate_adv_imgs(z)

        return adv_imgs

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch

        perturbation, adv_imgs = self.generate_adv_imgs(imgs)

        if optimizer_idx == 0:
            losses = self.generator_losses(labels, adv_imgs, perturbation, 'train')
            
            return losses["train_loss_generator"]

        if optimizer_idx == 1:
            losses = self.discriminator_loss(imgs, adv_imgs, 'train')
            
            return losses["train_loss_discriminator"]

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch

        perturbation, adv_imgs = self.generate_adv_imgs(imgs)
        losses = self.generator_losses(labels, adv_imgs, perturbation, 'validation')

        return imgs, labels, perturbation, adv_imgs

    def validation_epoch_end(self, outputs):
        imgs_batches, labels_batches, perturbation_batches, adv_imgs_batches = [torch.stack([output[i] for output in outputs])[:self.num_batches_to_log, :self.num_samples_to_log] for i in range(len(outputs[0]))]
        preds_batches = labels_batches

        wandb.log({
            "pred_imgs": [
                wandb.Image(
                    img,
                    caption=f'Pred: {pred}, Label: {label}'
                ) for imgs, labels, preds in zip(imgs_batches, labels_batches, preds_batches) for img, pred, label in zip(imgs, labels, preds)
            ],
            "pred_adv_imgs": [
                wandb.Image(
                    adv_img,
                    caption=f'Pred: {pred}, Label: {label}'
                ) for adv_imgs, labels, preds in zip(adv_imgs_batches, labels_batches, preds_batches) for adv_img, pred, label in zip(adv_imgs, labels, preds)
            ],
            "perturbation": [
                wandb.Image(
                    perturbation,
                    caption=f'Label: {label}'
                ) for labels, perturbations in zip(labels_batches, perturbation_batches) for label, perturbation in zip(labels, perturbations)
            ]
        })

    def loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def generate_adv_imgs(self, imgs):
        perturbation = self.generator(imgs)
        adv_imgs = torch.clamp(perturbation, -0.3, 0.3) + imgs
        adv_imgs = torch.clamp(adv_imgs, self.box_min, self.box_max)

        return perturbation, adv_imgs

    def generator_losses(self, labels, adv_imgs, perturbation, stage='train'):
        loss_generator_fake = self.generator_loss_fake(adv_imgs)
        loss_perturb = self.perturbation_loss(perturbation)
        loss_adv = self.target_model_loss(adv_imgs, labels)

        # Implementation from https://github.com/mathcbc/advGAN_pytorch
        # lossG = self.adv_lambda * loss_adv + self.pert_lambda * loss_perturb
        # My implementation
        loss_generator = (self.adv_lambda * loss_adv) + (self.gen_lambda * loss_generator_fake) + (self.pert_lambda * loss_perturb)

        losses = {
            f"{stage}_loss_generator_fake": loss_generator_fake,
            f"{stage}_loss_perturb": loss_perturb,
            f"{stage}_loss_adv": loss_adv,
            f"{stage}_loss_generator": loss_generator,
        }

        self.log_dict(
            losses,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )

        return losses 


    def generator_loss_fake(self, adv_imgs):
        pred_fake = self.discriminator(adv_imgs)
        valid = torch.ones_like(pred_fake, device=self.device)
        lossG_fake = self.loss(pred_fake, valid)

        return lossG_fake

    # Soft hinge loss to bound the magnitude of the perturbation
    def perturbation_loss(self, perturbation):
        # Implementation from https://github.com/mathcbc/advGAN_pytorch does this
        """
        norm_perturb = torch.norm(perturbation, 2, dim=1)
        loss_perturb = torch.mean(norm_perturb)
        """
        norm_perturb = torch.norm(perturbation, 2, dim=1)
        loss_perturb = torch.mean(torch.max(norm_perturb - self.C, torch.zeros(1, device=self.device)))

        return loss_perturb

    def target_model_loss(self, adv_imgs, labels):
        # Loss of fooling the target model:
        # Implementation from https://github.com/mathcbc/advGAN_pytorch
        preds = self.model(adv_imgs)
        probs = F.softmax(preds, dim=1)
        onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

        # C&W loss function

        # Probabilities of ground truth
        real = onehot_labels * probs
        real = torch.sum(real, dim=1)

        # Probabilities of the remaining classes
        # other, _ = torch.max((1 - onehot_labels) * probs - onehot_labels * 10000, dim=1)
        other = (1 - onehot_labels) * probs
        other, _ = torch.max(other, dim=1)

        zeros = torch.zeros_like(other)

        # If any other class than the ground truth was predicted the loss is zero
        # Otherwise the loss is the difference between
        # the prob of the ground truth and the second highest prob
        # In paper they do (other - real) which does not really makes sense to me
        loss_adv = torch.max(real - other, zeros)
        loss_adv = torch.sum(loss_adv)

        return loss_adv

    def discriminator_loss_real(self, imgs):
        pred_real = self.discriminator(imgs)
        valid = torch.ones_like(pred_real, device=self.device)
        lossD_real = self.loss(pred_real, valid)

        return lossD_real

    def discriminator_loss_fake(self, adv_imgs):
        pred_fake = self.discriminator(adv_imgs.detach())
        fake = torch.zeros_like(pred_fake, device=self.device)
        lossD_fake = self.loss(pred_fake, fake)

        return lossD_fake

    def discriminator_loss(self, imgs, adv_imgs, stage='train'):
        loss_real = self.discriminator_loss_real(imgs)
        loss_fake = self.discriminator_loss_fake(adv_imgs)

        loss_discriminator = (loss_real + loss_fake) / 2

        losses = {
            f"{stage}_loss_discriminator": loss_discriminator,
        }

        self.log_dict(
            losses,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )

        return losses

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        if self.current_epoch == 50:
            self.lr = 0.0001
        if self.current_epoch == 80:
            self.lr = 0.00001
