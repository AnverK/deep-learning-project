import argparse
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
from models.ape_gan.ape_gan import ApeGan
from attacks import FGSM, PGD

os.environ['WANDB_SAVE_CODE'] = "true"

pl.seed_everything(36)

os.makedirs(Config.LOGS_PATH, exist_ok=True)
os.makedirs(f'{Config.LOGS_PATH}/{Config.APE_GAN_FOLDER}/', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--adv-model", type=str, default='adv_gan')
args = parser.parse_args()

PathCreator = CreatePaths(args.adv_model)
TARGET_MODEL_PATH, ADV_MODEL_FOLDER, DEFENSE_MODEL_FOLDER = PathCreator.create_paths()

if args.adv_model == 'adv_gan':
    ADV_MODEL_PATH = f'{ADV_MODEL_FOLDER}/{Config.ADV_GAN_CKPT}'

    adv_model = AdvGAN.load_from_checkpoint(
        ADV_MODEL_PATH,
        is_distilled=Config.IS_DISTILLED,
        target_model_dir=TARGET_MODEL_PATH
    )

    adv_model.freeze()
    adv_model.eval()

elif args.adv_model == 'fgsm':
    adv_model = FGSM(target_model_dir=TARGET_MODEL_PATH)

elif args.adv_model == 'pgd':
    adv_model = PGD(target_model_dir=TARGET_MODEL_PATH)

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
    target_model_dir=TARGET_MODEL_PATH
)

wandb_logger = pl_loggers.WandbLogger(
    project='deep-learning',
    group='ape_gan',
    log_model=True,
    save_dir=Config.LOGS_PATH
)

checkpoint_callback = ModelCheckpoint(
    DEFENSE_MODEL_FOLDER,
    monitor="validation_accuracy_restored",
    save_top_k=1,
    save_last=True,
    filename='best',
    mode='max'
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
