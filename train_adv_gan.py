import os

import torch
from pl_bolts.datamodules import MNISTDataModule
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, ModelCheckpoint

from config import Config
from models.adv_gan.adv_gan import AdvGAN
from models.target_models.target_model import TargetModel

os.environ['WANDB_SAVE_CODE'] = "true"

pl.seed_everything(36)

os.makedirs(Config.LOGS_PATH, exist_ok=True)
os.makedirs(f'{Config.LOGS_PATH}/{Config.ADV_GAN_FOLDER}/', exist_ok=True)

dm = MNISTDataModule(
    f'{Config.LOGS_PATH}',
    batch_size=Config.ADV_GAN_BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    drop_last=True
)

model = AdvGAN(
    model_num_labels=10,
    image_nc=1,
    box_min=0,
    box_max=1,
    is_relativistic=False,
    is_blackbox=True,
    tensorflow=False,
    tf_target_model_dir=f'{Config.LOGS_PATH}/{Config.TARGET_MODEL_FOLDER}/adv_trained',
    target_model_dir=f'{Config.LOGS_PATH}/{Config.TARGET_MODEL_FOLDER}/converted_adv_trained/model.ckpt'
)

wandb_logger = pl_loggers.WandbLogger(
    project='deep-learning',
    group='adv_gan',
    log_model=True,
    save_dir=Config.LOGS_PATH
)
wandb_logger.watch(model)

checkpoint_callback = ModelCheckpoint(
    f'{Config.LOGS_PATH}/{Config.ADV_GAN_FOLDER}/',
    monitor="validation_accuracy_adversarial",
    save_top_k=1,
    save_last=True,
    mode='min'
)

callbacks = [checkpoint_callback]

trainer = Trainer(
    gpus=-1,
    max_epochs=200,
    precision=16,
    callbacks=callbacks,
    benchmark=True,
    num_sanity_val_steps=2,
    logger=wandb_logger,
)

trainer.fit(model, dm)
