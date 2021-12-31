import argparse
import os

import torch
from pl_bolts.datamodules import MNISTDataModule
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, ModelCheckpoint

from config import Config
from models.adv_gan.adv_gan import AdvGAN
from models.ape_gan.ape_gan import ApeGan
from attacks import FGSM, CW_L2

os.environ['WANDB_SAVE_CODE'] = "true"

pl.seed_everything(36)

os.makedirs(Config.LOGS_PATH, exist_ok=True)
os.makedirs(f'{Config.LOGS_PATH}/{Config.APE_GAN_FOLDER}/', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--attack", type=str, default='adv_gan_blackbox')
parser.add_argument("--adv-model-path", type=str, default=f'{Config.LOGS_PATH}/{Config.ADV_GAN_FOLDER}/last.ckpt')
parser.add_argument("--robust-model-path", type=str,
                    default=f'{Config.LOGS_PATH}/{Config.TARGET_MODEL_FOLDER}/converted_adv_trained/{Config.TARGET_MODEL_CKPT}')
parser.add_argument("--defense-model-path", type=str,
                    default=f'{Config.LOGS_PATH}/{Config.APE_GAN_FOLDER}/')
args = parser.parse_args()

defense_model_path = args.defense_model_path
adv_model_path = args.adv_model_path

if args.attack == 'adv_gan_whitebox':
    defense_model_path = f'{Config.LOGS_PATH}/{Config.APE_GAN_FOLDER}/whitebox/'

    adv_model_path = f'{Config.LOGS_PATH}/{Config.ADV_GAN_FOLDER}/adv_gan_whitebox/last.ckpt'

    attack = AdvGAN.load_from_checkpoint(
        adv_model_path,
        model_num_labels=10,
        image_nc=1,
        box_min=0,
        box_max=1,
        tensorflow=False,
        is_blackbox=False,
        is_relativistic=False,
        target_model_dir=args.robust_model_path
    )

    attack.freeze()
    attack.eval()
elif args.attack == 'adv_gan_blackbox':
    defense_model_path = f'{Config.LOGS_PATH}/{Config.APE_GAN_FOLDER}/adv_gan_blackbox'

    adv_model_path = f'{Config.LOGS_PATH}/{Config.ADV_GAN_FOLDER}/blackbox/last.ckpt'

    attack = AdvGAN.load_from_checkpoint(
        adv_model_path,
        model_num_labels=10,
        image_nc=1,
        box_min=0,
        box_max=1,
        tensorflow=False,
        is_blackbox=True,
        is_relativistic=False,
        target_model_dir=args.robust_model_path
    )

    attack.freeze()
    attack.eval()
elif args.attack == 'fgsm':
    defense_model_path = f'{Config.LOGS_PATH}/{Config.APE_GAN_FOLDER}/fgsm'

    attack = FGSM(target_model_dir=args.robust_model_path)
elif args.attack == 'cw_l2':
    defense_model_path = f'{Config.LOGS_PATH}/{Config.APE_GAN_FOLDER}/cw_l2'

    attack = CW_L2(target_model_dir=args.robust_model_path)
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
    attack=attack,
    target_model_dir=args.robust_model_path
)

wandb_logger = pl_loggers.WandbLogger(
    project='deep-learning',
    group='ape_gan',
    log_model=True,
    save_dir=Config.LOGS_PATH
)

checkpoint_callback = ModelCheckpoint(
    defense_model_path,
    monitor="validation_accuracy_adversarial",
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
    num_sanity_val_steps = 2,
    logger=wandb_logger,
)

trainer.fit(model, dm)