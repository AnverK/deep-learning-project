import argparse
import os

import torch
from pl_bolts.datamodules import MNISTDataModule, CIFAR10DataModule
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, ModelCheckpoint

from config import Config
from create_paths import CreatePaths
from models.adv_gan.adv_gan import AdvGAN
from models.ape_gan.ape_gan import ApeGan
from attacks import FGSM, PGD
from models.target_models.target_model import ModelCIFAR10, TargetModelCIFAR10
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

os.environ['WANDB_SAVE_CODE'] = "true"

pl.seed_everything(36)

os.makedirs(Config.LOGS_PATH, exist_ok=True)
os.makedirs(f'{Config.LOGS_PATH}/{Config.APE_GAN_FOLDER}/{Config.DATASET}', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--adv-model", type=str, default='fgsm')
args = parser.parse_args()

PathCreator = CreatePaths(args.adv_model)
TARGET_MODEL_PATH, ADV_MODEL_FOLDER, DEFENSE_MODEL_FOLDER = PathCreator.create_paths()
DEFENSE_MODEL_FOLDER = f"{DEFENSE_MODEL_FOLDER}/{Config.DATASET}"

with tf.device('/GPU:0'):
    tf_model = ModelCIFAR10('eval')
    model_file = tf.train.latest_checkpoint('cifar10_challenge/models/adv_trained')
    saver = tf.train.Saver()

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)
saver.restore(sess, model_file)

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
robust_model = TargetModelCIFAR10(tf_model, sess).to(device)

if args.adv_model == 'fgsm':
    adv_model = FGSM(eps=8 / 255, target_model=robust_model)
elif args.adv_model == 'pgd':
    adv_model = PGD(eps=8 / 255, target_model=robust_model)
else:
    print("This attack is not implemented!")
    quit()

dm = CIFAR10DataModule(
    f'{Config.LOGS_PATH}',
    batch_size=Config.APE_GAN_BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    drop_last=True
)

model = ApeGan(
    3,
    Config.APE_GAN_gen_loss_scale,
    Config.APE_GAN_dis_loss_scale,
    Config.APE_GAN_lr,
    attack=adv_model,
    target_model=robust_model
)

wandb_logger = pl_loggers.WandbLogger(
    project='deep-learning',
    group='ape_gan_cifar10',
    name=f'{args.adv_model}-{"blackbox" if Config.IS_BLACK_BOX else "whitebox"}',
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
