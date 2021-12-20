from discriminator import Discriminator
from generator import Generator
from targetModel import TargetModel

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


# custom weights initialization called on generator and discriminator
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
            **kwargs
    ):
        super().__init__()
        self.b2 = b2
        self.b1 = b1
        self.lr = lr
        self.save_hyperparameters()

        output_nc = image_nc
        self.model_num_labels = model_num_labels
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max

        # networks
        self.gen_input_nc = image_nc
        self.generator = Generator(self.gen_input_nc, image_nc)
        self.discriminator = Discriminator(image_nc)

        self.validation_z = torch.randn(8, self.gen_input_nc)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.model = TargetModel.load_from_checkpoint(checkpoint_path="target.ckpt")

    def forward(self, z):
        return self.generator(z)

    def loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch

        perturbation = self.generator(imgs)
        # add a clipping trick
        adv_images = torch.clamp(perturbation, -0.3, 0.3) + imgs
        adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

        # optimize G
        if optimizer_idx == 0:
            # cal G's loss in GAN
            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            pred_fake = self.discriminator(adv_images)
            valid = torch.ones_like(pred_fake)
            valid = valid.type_as(imgs)
            lossG_fake = self.loss(pred_fake, valid)
            self.log("Generator loss", lossG_fake)

            # calculate perturbation norm
            # C = 0.1
            norm_perturb = torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1)
            loss_perturb = torch.mean(norm_perturb)
            self.log("Perturbation loss", loss_perturb)
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # cal adv loss
            preds = self.model(adv_images)
            probs = F.softmax(preds, dim=1)
            onehot_labels = torch.eye(self.model_num_labels)[labels]
            onehot_labels = onehot_labels.type_as(onehot_labels)
            # C&W loss function
            real = torch.sum(onehot_labels * probs, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(real - other, zeros)
            loss_adv = torch.sum(loss_adv)
            self.log("Adversary loss", loss_adv)

            adv_lambda = 10
            pert_lambda = 1
            lossG = adv_lambda * lossG_fake + loss_adv + pert_lambda * loss_perturb
            self.log("Generator loss Overall", lossG)
            tqdm_dict = {"lossG": self.lossG}
            output = OrderedDict({"loss": lossG, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        if optimizer_idx == 1:
            # how well can it label as real?
            pred_real = self.discriminator(imgs)
            valid = torch.ones_like(pred_real)
            valid = valid.type_as(imgs)
            lossD_real = self.loss(pred_real, valid)

            # how well can it label as fake?
            pred_fake = self.discriminator(adv_images.detach())
            fake = torch.zeros_like(pred_fake)
            fake = fake.type_as(imgs)
            lossD_fake = self.loss(pred_fake, fake)

            lossD = (lossD_fake + lossD_real) / 2
            self.log("Discriminator loss", lossD)
            tqdm_dict = {"lossD": lossD}
            output = OrderedDict({"loss": lossD, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

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
