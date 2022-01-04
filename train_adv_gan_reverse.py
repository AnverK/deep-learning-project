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
from models.adv_gan.adv_gan_reverse import AdvGAN as AdvGANReverse
from models.target_models.target_model import TargetModel
from attacks import FGSM, PGD

os.environ['WANDB_SAVE_CODE'] = "true"

pl.seed_everything(36)

os.makedirs(Config.LOGS_PATH, exist_ok=True)
os.makedirs(f'{Config.LOGS_PATH}/{Config.ADV_GAN_FOLDER}_reversed/', exist_ok=True)

dm = MNISTDataModule(
    f'{Config.LOGS_PATH}',
    batch_size=Config.ADV_GAN_BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    drop_last=True
)

parser = argparse.ArgumentParser()
parser.add_argument("--adv-model", type=str, default='adv_gan')
parser.add_argument("--attack-is-distilled", default=False, action='store_true')
parser.add_argument("--defense-is-distilled", default=False, action='store_true')
args = parser.parse_args()

PathCreator = CreatePaths(args.adv_model, is_distilled=args.attack_is_distilled)
TARGET_MODEL_PATH, ADV_MODEL_FOLDER, DEFENSE_MODEL_FOLDER = PathCreator.create_paths()

if args.adv_model == 'adv_gan':
    ADV_MODEL_PATH = f'{ADV_MODEL_FOLDER}/{Config.ADV_GAN_CKPT}'

    adv_model = AdvGAN.load_from_checkpoint(
        ADV_MODEL_PATH,
        is_distilled=args.attack_is_distilled,
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

model = AdvGANReverse(
    is_distilled=args.defense_is_distilled,
    target_model_dir=TARGET_MODEL_PATH,
    attack=adv_model
)

wandb_logger = pl_loggers.WandbLogger(
    project='deep-learning',
    group='adv_gan_reverse',
    name=f'{args.adv_model}{"-blackbox" if Config.IS_BLACK_BOX else "-whitebox"}{"-attack_is_distilled" if args.attack_is_distilled else ""}{"-defense_is_distilled" if args.defense_is_distilled else ""}',
    log_model=True,
    save_dir=Config.LOGS_PATH
)

checkpoint_callback = ModelCheckpoint(
    f'{Config.LOGS_PATH}/adv_gan_reversed/{args.adv_model}{"-blackbox" if Config.IS_BLACK_BOX else "-whitebox"}{"-attack_is_distilled" if args.attack_is_distilled else ""}{"-defense_is_distilled" if args.defense_is_distilled else ""}',
    monitor="validation_accuracy_restored",
    save_top_k=1,
    filename='best',
    save_last=True,
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
