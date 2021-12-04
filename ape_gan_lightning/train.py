# -*- coding: utf-8 -*-

import os
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer

import matplotlib.pyplot as plt

from models import Generator, Discriminator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PreparedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_data = None
        self.example_data = None

    def setup(self, stage=None):
        train_data = torch.load(self.data_dir, map_location=device)
        self.train_data = TensorDataset(train_data["normal"], train_data["adv"])
        self.example_data = TensorDataset(train_data["normal"], train_data["adv"])[:5]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)


class ApeGan(pl.LightningModule):
    def __init__(self, in_ch, xi1, xi2, lr, checkpoint_path):
        super().__init__()
        self.xi1 = xi1
        self.xi2 = xi2
        self.lr = lr
        self.checkpoint_path = checkpoint_path
        self.generator = Generator(in_ch).to(device)
        self.discriminator = Discriminator(in_ch).to(device)
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        x, x_adv = batch

        current_size = x.size(0)
        x, x_adv = Variable(x.to(device)), Variable(x_adv.to(device))

        loss_bce = nn.BCELoss()
        loss_mse = nn.MSELoss()

        t_real = Variable(torch.ones(current_size).to(device))
        t_fake = Variable(torch.zeros(current_size).to(device))

        # Train discriminator
        opt_d, opt_g = self.optimizers()
        y_real = self.discriminator(x).squeeze()
        x_fake = self.generator(x_adv)
        y_fake = self.discriminator(x_fake).squeeze()

        loss_d = loss_bce(y_real, t_real) + loss_bce(y_fake, t_fake)
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()

        # Train generator
        for _ in range(2):
            x_fake = self.generator(x_adv)
            y_fake = self.discriminator(x_fake).squeeze()

            loss_g = self.xi1 * loss_mse(x_fake, x) + self.xi2 * loss_bce(y_fake, t_real)
            opt_g.zero_grad()
            self.manual_backward(loss_g)
            opt_g.step()

    def configure_optimizers(self):
        lr = self.lr
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        return opt_d, opt_g

    def on_epoch_end(self):
        x, x_adv = self.trainer.datamodule.example_data
        x, x_adv = x.to(device), x_adv.to(device)
        self.generator.eval()
        with torch.no_grad():
            x_fake = self.generator(Variable(x_adv.to(device))).data
        ApeGan.show_images(self.current_epoch, x, x_adv, x_fake, self.checkpoint_path)
        self.generator.train()
        torch.save({"generator": self.generator.state_dict(),
                    "discriminator": self.discriminator.state_dict()},
                   os.path.join(self.checkpoint_path, "{}.tar".format(self.current_epoch + 1)))

    @staticmethod
    def show_images(e, x, x_adv, x_fake, save_dir):
        fig, axes = plt.subplots(3, 5, figsize=(10, 6))
        for i in range(5):
            axes[0, i].axis("off"), axes[1, i].axis("off"), axes[2, i].axis("off")
            axes[0, i].imshow(x[i].cpu().numpy().transpose((1, 2, 0)))
            # axes[0, i].imshow(x[i, 0].cpu().numpy(), cmap="gray")
            axes[0, i].set_title("Normal")

            axes[1, i].imshow(x_adv[i].cpu().numpy().transpose((1, 2, 0)))
            # axes[1, i].imshow(x_adv[i, 0].cpu().numpy(), cmap="gray")
            axes[1, i].set_title("Adv")

            axes[2, i].imshow(x_fake[i].cpu().numpy().transpose((1, 2, 0)))
            # axes[2, i].imshow(x_fake[i, 0].cpu().numpy(), cmap="gray")
            axes[2, i].set_title("APE-GAN")
        plt.axis("off")
        plt.savefig(os.path.join(save_dir, "result_{}.png".format(e)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adv-data-path", type=str, default="./data.tar")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--xi1", type=float, default=0.7)
    parser.add_argument("--xi2", type=float, default=0.3)
    parser.add_argument("--checkpoint", type=str, default="./checkpoint")

    args = parser.parse_args()
    dm = PreparedDataModule(data_dir=args.adv_data_path, batch_size=128)
    model = ApeGan(in_ch=1, xi1=0.7, xi2=0.3, lr=0.0002, checkpoint_path=args.checkpoint)
    accelerator = None if torch.cuda.is_available() else 'cpu'
    gpus = 1 if torch.cuda.is_available() else 0
    trainer = Trainer(gpus=gpus, accelerator=accelerator, max_epochs=args.epochs, progress_bar_refresh_rate=20)
    trainer.fit(model, dm)
