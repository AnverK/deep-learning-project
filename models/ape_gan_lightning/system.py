import pytorch_lightning as pl
import torch

from .models import MnistCNN, CifarCNN, Generator, Discriminator

class ApeGan(pl.LightningModule):
    def __init__(self, in_ch, xi1, xi2, lr, checkpoint_path):
        super().__init__()
        self.xi1 = xi1
        self.xi2 = xi2
        self.lr = lr
        self.checkpoint_path = checkpoint_path
        self.generator = Generator(in_ch)
        self.discriminator = Discriminator(in_ch)
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        X, X_adv = batch

        loss_bce = nn.BCELoss()
        loss_mse = nn.MSELoss()

        t_real = torch.ones(X.shape[0], device=self.device)
        t_fake = torch.zeros(X.shape[0], device=self.device)

        # Train discriminator
        opt_d, opt_g = self.optimizers()
        y_real = self.discriminator(X).squeeze()
        x_fake = self.generator(X_adv)
        y_fake = self.discriminator(x_fake).squeeze()

        loss_d = loss_bce(y_real, t_real) + loss_bce(y_fake, t_fake)
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()

        # Train generator
        for _ in range(2):
            x_fake = self.generator(X_adv)
            y_fake = self.discriminator(x_fake).squeeze()

            loss_g = self.xi1 * loss_mse(x_fake, X) + self.xi2 * loss_bce(y_fake, t_real)
            opt_g.zero_grad()
            self.manual_backward(loss_g)
            opt_g.step()

    def configure_optimizers(self):
        lr = self.lr
        
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        return opt_d, opt_g