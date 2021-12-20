import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import wandb


class TargetModel(LightningModule):
    def __init__(
            self,
            lr: float = 0.001,
            num_samples_to_log = 16,
            num_batches_to_log = 1,
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

        self.num_samples_to_log = num_samples_to_log
        self.num_batches_to_log = num_batches_to_log

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
        preds, loss = self.shared_step(train_batch)
        
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        preds, loss = self.shared_step(val_batch)
        
        self.log('val_loss', loss)

        return val_batch, preds

    def validation_epoch_end(self, outputs):
        batches = [output[0] for output in outputs]
        imgs_batches = torch.stack([batch[0] for batch in batches])[:self.num_batches_to_log, :self.num_samples_to_log]
        labels_batches = torch.stack([batch[1] for batch in batches])[:self.num_batches_to_log, :self.num_samples_to_log]
        preds_batches = torch.stack([output[1] for output in outputs])[:self.num_batches_to_log, :self.num_samples_to_log].argmax(-1)

        wandb.log({
            "pred": [
                wandb.Image(
                    img,
                    caption=f'Pred: {pred}, Label: {label}'
                ) for imgs, labels, preds in zip(imgs_batches, labels_batches, preds_batches) for img, pred, label in zip(imgs, labels, preds)
            ]
        })

    def shared_step(self, batch):
        imgs, labels = batch

        preds = torch.zeros(labels.size(0), 10, device=self.device)
        preds = self.input_net(imgs)
        preds = preds.view(-1, 64 * 4 * 4)
        preds = self.output_net(preds)
        
        loss = self.loss(preds, labels)

        return preds, loss
