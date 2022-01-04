import argparse
import os

import torch
from pl_bolts.datamodules import MNISTDataModule
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from config import Config
from create_paths import CreatePaths
from models.adv_gan.adv_gan import AdvGAN

os.environ['WANDB_SAVE_CODE'] = "true"

pl.seed_everything(36)

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is-blackbox", default=False, action='store_true')
    parser.add_argument("--is-distilled", default=False, action='store_true')
    parser.add_argument("--no-wandb", default=False, action='store_true')

    args = parser.parse_args()

    if args.no_wandb:
        os.environ['WANDB_MODE'] = "disabled"
    os.makedirs(Config.LOGS_PATH, exist_ok=True)
    os.makedirs(f'{Config.LOGS_PATH}/{Config.ADV_GAN_FOLDER}/', exist_ok=True)

    PathCreator = CreatePaths(adv_model='adv_gan', is_blackbox=args.is_blackbox, is_distilled=args.is_distilled)
    TARGET_MODEL_PATH, ADV_MODEL_FOLDER, _ = PathCreator.create_paths()

    dm = MNISTDataModule(
        f'{Config.LOGS_PATH}',
        batch_size=Config.ADV_GAN_BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        drop_last=True
    )

    model = AdvGAN(
        is_distilled=args.is_distilled,
        target_model_dir=TARGET_MODEL_PATH
    )

    wandb_logger = pl_loggers.WandbLogger(
        project='deep-learning',
        group='adv_gan',
        name=f'{"blackbox" if args.is_blackbox else "whitebox"}{"-distilled" if args.is_distilled else ""}',
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

    if device.type == 'cpu':
        gpus = None
        precision = 32
    else:
        gpus = -1
        precision = 16

    trainer = Trainer(
        gpus=gpus,
        max_epochs=50,
        precision=precision,
        callbacks=callbacks,
        benchmark=True,
        num_sanity_val_steps=2,
        logger=wandb_logger,
    )

    trainer.fit(model, dm)
