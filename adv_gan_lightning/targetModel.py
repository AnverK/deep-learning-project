import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


class TargetModel(LightningModule):
    def __init__(
            self,
            lr: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.output_net = nn.Sequential(
            nn.Linear(64 * 4 * 4, 200),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        )

    def forward(self, x):
        x = self.input_net(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.output_net(x)
        return x

    def configure_optimizers(self):
        lr = self.hparams.lr

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def training_step(self, train_batch, batch_idx):
        imgs, labels = train_batch

        preds = torch.zeros(labels.size(0), 10)
        preds = preds.type_as(labels)
        preds = self.input_net(imgs)
        preds = preds.view(-1, 64 * 4 * 4)
        preds = self.output_net(preds)
        loss = self.loss(preds, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        imgs, labels = val_batch

        preds = torch.zeros(labels.size(0), 10)
        preds = preds.type_as(labels)
        preds = self.input_net(imgs)
        preds = preds.view(-1, 64 * 4 * 4)
        preds = self.output_net(preds)
        loss = self.loss(preds, labels)
        self.log('val_loss', loss)
