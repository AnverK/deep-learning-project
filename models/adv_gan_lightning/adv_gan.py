from .discriminator import Discriminator
from .generator import Generator
from .target_model import TargetModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

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
        self.adv_lambda = 10
        self.pert_lambda = 1

    def forward(self, z):
        return self.generator(z)

    def loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch

        perturbation = self.generator(imgs)
        adv_images = torch.clamp(perturbation, -0.3, 0.3) + imgs
        adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

        if optimizer_idx == 0:
            pred_fake = self.discriminator(adv_images)
            
            valid = torch.ones_like(pred_fake, device=self.device)
            
            lossG_fake = self.loss(pred_fake, valid)

            # Soft hinge loss to bound the magnitude of the perturbation
            # Implementation from https://github.com/mathcbc/advGAN_pytorch does this
            '''
            norm_perturb = torch.norm(perturbation, 2, dim=1)
            loss_perturb = torch.mean(norm_perturb)
            '''
            # Paper does this
            norm_perturb = torch.norm(perturbation, 2)
            loss_perturb = torch.max(norm_perturb - self.C, torch.zeros(1, device=self.device))

            # Loss of fooling the target model:
            # Implementation from https://github.com/mathcbc/advGAN_pytorch
            preds = self.model(adv_images)
            probs = F.softmax(preds, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

            # C&W loss function
            real = onehot_labels * probs
            real = torch.max(real, dim=1)

            # other, _ = torch.max((1 - onehot_labels) * probs - onehot_labels * 10000, dim=1)
            other = (1 - onehot_labels) * probs
            other, _ = torch.max(other, dim=1)

            zeros = torch.zeros_like(other)

            # If any other class than the ground truth was predicted the loss is zero
            # Otherwise the loss is the difference between
            # the prob of the ground truth and the second highest prob
            loss_adv = torch.max(real - other, zeros)
            loss_adv = torch.sum(loss_adv)

            lossG = self.adv_lambda * loss_adv + self.pert_lambda * loss_perturb
            
            losses = {
                "lossG_fake": lossG_fake,
                "loss_perturb": loss_perturb,
                "loss_adv": loss_adv,
                "lossG": lossG,
            }

            self.log_dict(
                losses,
                prog_bar=True,
                on_step=True,
                on_epoch=True
            )
            
            return lossG

        if optimizer_idx == 1:
            loss_discriminator = self.discriminator_loss(imgs, adv_images)
            
            losses = {
                "loss_discriminator": loss_discriminator,
            }

            self.log_dict(
                losses,
                prog_bar=True,
                on_step=True,
                on_epoch=True
            )

            return loss_discriminator

    def discriminator_loss_real(self, imgs):
        pred_real = self.discriminator(imgs)
        valid = torch.ones_like(pred_real, device=self.device)
        lossD_real = self.loss(pred_real, valid)

        return lossD_real

    def discriminator_loss_fake(self, imgs, adv_images):
        pred_fake = self.discriminator(adv_images.detach())
        fake = torch.zeros_like(pred_fake, device=self.device)
        lossD_fake = self.loss(pred_fake, fake)

        return lossD_fake

    def discriminator_loss(self, imgs, adv_images):
        lossD_real = self.discriminator_loss_real(imgs)
        lossD_fake = self.discriminator_loss_fake(imgs, adv_images)

        loss_discriminator = (lossD_fake + lossD_real) / 2

        return loss_discriminator

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
