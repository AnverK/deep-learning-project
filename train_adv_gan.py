import os

import torch
from pl_bolts.datamodules import MNISTDataModule
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, ModelCheckpoint

from config import Config
from create_paths import CreatePaths
from models.adv_gan.adv_gan import AdvGAN
from models.target_models.target_model import TargetModel

os.environ['WANDB_SAVE_CODE'] = "true"

pl.seed_everything(36)

os.makedirs(Config.LOGS_PATH, exist_ok=True)
os.makedirs(f'{Config.LOGS_PATH}/{Config.ADV_GAN_FOLDER}/', exist_ok=True)

PathCreator = CreatePaths()
TARGET_MODEL_PATH, ADV_MODEL_FOLDER, _ = PathCreator.create_paths()

dm = MNISTDataModule(
    f'{Config.LOGS_PATH}',
    batch_size=Config.ADV_GAN_BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    drop_last=True
)

model = AdvGAN(
    is_distilled=Config.IS_DISTILLED,
    target_model_dir=TARGET_MODEL_PATH
)

wandb_logger = pl_loggers.WandbLogger(
    project='deep-learning',
    group='adv_gan',
    name=f'{"blackbox" if Config.IS_BLACK_BOX else "whitebox"}{"-distilled" if Config.IS_DISTILLED else ""}',
    log_model=True,
    save_dir=Config.LOGS_PATH
)

checkpoint_callback = ModelCheckpoint(
    ADV_MODEL_FOLDER,
    monitor="validation_accuracy_adversarial",
    save_top_k=1,
    filename='best',
    save_last=True,
    mode='min'
)

callbacks = [checkpoint_callback]

trainer = Trainer(
    gpus=-1,
    max_epochs=50,
    precision=16,
    callbacks=callbacks,
    benchmark=True,
    num_sanity_val_steps=2,
    logger=wandb_logger,
)

trainer.fit(model, dm)
