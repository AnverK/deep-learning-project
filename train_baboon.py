from pl_bolts.datamodules import MNISTDataModule
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, ModelCheckpoint
import os

os.environ['WANDB_SAVE_CODE'] = "true"

from models.ape_gan_lightning.ape_gan import ApeGan
from models.adv_gan_lightning.adv_gan import AdvGAN
from models.combined.baboon import Baboon
from config import Config

pl.seed_everything(36)

os.makedirs(Config.LOGS_PATH, exist_ok=True)
os.makedirs(f'{Config.LOGS_PATH}/{Config.APE_GAN_FOLDER}/', exist_ok=True)

dm = MNISTDataModule(
    f'{Config.LOGS_PATH}',
    batch_size=Config.APE_GAN_BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    drop_last=True
)

adv_gan = AdvGAN(
    model_num_labels=10,
    image_nc=1,
    box_min=0,
    box_max=1,
    target_model_dir=f'{Config.LOGS_PATH}/{Config.TARGET_MODEL_FOLDER}/last.ckpt'
)

ape_gan = ApeGan(
    1,
    Config.APE_GAN_gen_loss_scale,
    Config.APE_GAN_dis_loss_scale,
    Config.APE_GAN_lr,
    attack=adv_gan,
    target_model_dir=f'{Config.LOGS_PATH}/{Config.TARGET_MODEL_FOLDER}/last.ckpt'
)

model = Baboon(
    adv_gan,
    ape_gan
)

wandb_logger = pl_loggers.WandbLogger(
    project='deep-learning',
    group='baboon',
    log_model=True,
    save_dir=Config.LOGS_PATH
)
wandb_logger.watch(model)

checkpoint_callback = ModelCheckpoint(
    f'{Config.LOGS_PATH}/{Config.BABOON_FOLDER}',
    monitor="validation_loss_generator",
    save_top_k=1,
    save_last=True,
    mode='min'
)

callbacks = []

trainer = Trainer(
    gpus=-1,
    max_epochs=60,
    precision=16,
    callbacks=callbacks,
    benchmark=True,
    num_sanity_val_steps=2,
    logger=wandb_logger,
)

trainer.fit(model, dm)
