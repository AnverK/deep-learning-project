import argparse
import os

import torch
from torchvision import datasets

from config import Config
from create_paths import CreatePaths
from models.adv_gan.adv_gan import AdvGAN
from models.ape_gan.ape_gan import ApeGan
from models.target_models.target_model import TargetModel
from attacks import FGSM, PGD

import pytorch_lightning as pl

pl.seed_everything(51)


def check_distance(X, X_adv, eps=0.3):
    norm = torch.max(torch.abs(X - X_adv))
    print(norm)
    return norm <= eps + 1e-5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adv-model", type=str, default='adv_gan')
    parser.add_argument("--attack", type=str, default='')
    parser.add_argument("--no-eval-defense", default=False, action='store_true')
    parser.add_argument("--attack-is-distilled", default=False, action='store_true')
    parser.add_argument("--defense-is-distilled", default=False, action='store_true')

    # currently not used TODO
    parser.add_argument("--dataset", type=str, default='mnist')
    parser.add_argument("--eps", type=float, default=0.3)
    args = parser.parse_args()

    if args.attack == '':
        args.attack = args.adv_model

    PathCreator = CreatePaths(args.adv_model, is_distilled=args.defense_is_distilled)
    TARGET_MODEL_PATH, ADV_MODEL_FOLDER, DEFENSE_MODEL_FOLDER = PathCreator.create_paths()

    ROBUST_MODEL_PATH = f'{Config.LOGS_PATH}/{Config.TARGET_MODEL_FOLDER}/converted_secret/model.ckpt'

    if args.adv_model == 'adv_gan':
        ADV_MODEL_PATH = f'{ADV_MODEL_FOLDER}/{Config.ADV_GAN_CKPT}'

        adv_model = AdvGAN.load_from_checkpoint(ADV_MODEL_PATH,
                                                is_distilled=args.defense_is_distilled,
                                                target_model_dir=TARGET_MODEL_PATH
                                                )

        adv_model.freeze()
        adv_model.eval()

    elif args.adv_model == 'fgsm':
        adv_model = FGSM(target_model_dir=TARGET_MODEL_PATH)

    elif args.adv_model == 'pgd':
        adv_model = PGD(target_model_dir=TARGET_MODEL_PATH)

    elif args.adv_model != '':
        print("This attack is not implemented!")
        quit()

    eval_defense = not args.no_eval_defense
    if eval_defense:
        DEFENSE_MODEL_PATH = f'{DEFENSE_MODEL_FOLDER}/{Config.APE_GAN_CKPT}'
        defense_model = ApeGan.load_from_checkpoint(DEFENSE_MODEL_PATH,
                                                    in_ch=1,
                                                    gen_loss_scale=Config.APE_GAN_gen_loss_scale,
                                                    dis_loss_scale=Config.APE_GAN_dis_loss_scale,
                                                    lr=Config.APE_GAN_lr,
                                                    attack=adv_model,
                                                    target_model_dir=TARGET_MODEL_PATH
                                                    )

    if args.dataset == 'mnist':
        test_data = datasets.MNIST('mnist', train=False, download=True)
    else:
        raise Exception("Other datasets are not supported yet")

    PathCreator = CreatePaths(
        adv_model=args.attack,
        is_blackbox=Config.IS_BLACK_BOX,
        is_distilled=args.attack_is_distilled)
    TARGET_MODEL_PATH, ADV_MODEL_FOLDER, DEFENSE_MODEL_FOLDER_ATTACK = PathCreator.create_paths()

    if args.attack == 'adv_gan':
        ADV_MODEL_PATH = f'{ADV_MODEL_FOLDER}/{Config.ADV_GAN_CKPT}'

        attack = AdvGAN.load_from_checkpoint(ADV_MODEL_PATH,
                                             is_distilled=args.attack_is_distilled,
                                             target_model_dir=TARGET_MODEL_PATH
                                             )

        attack.freeze()
        attack.eval()

    elif args.attack == 'fgsm':
        attack = FGSM(target_model_dir=TARGET_MODEL_PATH)

    elif args.attack == 'pgd':
        attack = PGD(target_model_dir=TARGET_MODEL_PATH)

    elif args.attack != '':
        print("This attack is not implemented!")
        quit()

    X = test_data.data
    X = X / 255
    X = torch.unsqueeze(X, 1)

    X_adv = attack(X)

    y = test_data.targets
    if eval_defense:
        X_res = defense_model(X_adv)

    if not check_distance(X, X_adv, eps=args.eps):
        raise Exception("Adversarial samples has too large distance from original samples")

    robust_model = TargetModel()
    robust_model.load_state_dict(torch.load(ROBUST_MODEL_PATH))
    robust_model.eval()

    print(f"Adversarial examples are generated from {os.path.basename(os.path.normpath(DEFENSE_MODEL_FOLDER_ATTACK))}")
    print(f"APE-GAN was trained on {os.path.basename(os.path.normpath(DEFENSE_MODEL_FOLDER))} \n")

    with torch.no_grad():
        probs = robust_model(X_adv)
        pred = torch.argmax(probs, dim=1)
        accuracy = torch.sum(pred == y) / len(y)
        print(f"Accuracy on adversarial samples is {accuracy.item()}")

        if eval_defense:
            probs = robust_model(X_res)
            pred = torch.argmax(probs, dim=1)
            accuracy = torch.sum(pred == y) / len(y)
            print(f"Accuracy on restored samples is {accuracy.item()}")
