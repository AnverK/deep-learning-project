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
from models.ape_gan.ape_gan import ApeGan
from attacks import FGSM, PGD

os.environ['WANDB_SAVE_CODE'] = "true"

pl.seed_everything(36)
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adv-model", type=str, default='adv_gan')
    parser.add_argument("--is-blackbox", default=False, action='store_true')
    parser.add_argument("--is-distilled", default=False, action='store_true')

    args = parser.parse_args()

    os.makedirs(Config.LOGS_PATH, exist_ok=True)
    os.makedirs(f'{Config.LOGS_PATH}/{Config.APE_GAN_FOLDER}/', exist_ok=True)

    PathCreator = CreatePaths(adv_model='adv_gan', is_blackbox=args.is_blackbox, is_distilled=args.is_distilled)
    target_model_path, adv_model_folder, defense_model_folder = PathCreator.create_paths()

    if args.adv_model == 'adv_gan':
        ADV_MODEL_PATH = f'{adv_model_folder}/{Config.ADV_GAN_CKPT}'

        adv_model = AdvGAN.load_from_checkpoint(ADV_MODEL_PATH)
        adv_model.freeze()
    elif args.adv_model == 'fgsm':
        adv_model = FGSM(target_model_dir=target_model_path)
    elif args.adv_model == 'pgd':
        adv_model = PGD(target_model_dir=target_model_path)
    else:
        print("This attack is not implemented!")
        quit()

    dm = MNISTDataModule(
        f'{Config.LOGS_PATH}',
        batch_size=Config.APE_GAN_BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        drop_last=True
    )

    model = ApeGan(
        1,
        Config.APE_GAN_gen_loss_scale,
        Config.APE_GAN_dis_loss_scale,
        Config.APE_GAN_lr,
        attack=adv_model,
        target_model_dir=target_model_path
    )

    wandb_logger = pl_loggers.WandbLogger(
        project='deep-learning',
        group='ape_gan',
        name=f'{args.adv_model}-{"blackbox" if args.is_blackbox else "whitebox"}{"-distilled" if args.is_distilled else ""}',
        log_model=True,
        save_dir=Config.LOGS_PATH
    )

    checkpoint_callback = ModelCheckpoint(
        defense_model_folder,
        monitor="validation_accuracy_restored",
        save_top_k=1,
        save_last=True,
        filename='best',
        mode='max'
    )

    callbacks = [checkpoint_callback]

    if device == 'cpu':
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
