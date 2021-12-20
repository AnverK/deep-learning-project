from targetModel import TargetModel

from pl_bolts.datamodules import MNISTDataModule
from pytorch_lightning import Trainer
import torch

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = 2

dm = MNISTDataModule('.', batch_size=BATCH_SIZE)  # if I set num_workers I get an Error
model = TargetModel()
trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=40, progress_bar_refresh_rate=20)
trainer.fit(model, dm)
trainer.save_checkpoint("target.ckpt")