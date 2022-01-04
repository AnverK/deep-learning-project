import argparse

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

ROBUST_MODEL_PATH = f'{Config.LOGS_PATH}/{Config.TARGET_MODEL_FOLDER}/converted_secret/model.ckpt'


def check_distance(X, X_adv, eps=0.3):
    norm = torch.max(torch.abs(X - X_adv))
    print(norm)
    return norm <= eps + 1e-5


def load_attack(adv_model_type, target_model_folder=None, adv_model_folder=None):
    if adv_model_type == 'adv_gan':
        adv_model_path = f'{adv_model_folder}/{Config.ADV_GAN_CKPT}'

        adv_model = AdvGAN.load_from_checkpoint(adv_model_path)

        adv_model.freeze()
        adv_model.eval()

    elif adv_model_type == 'fgsm':
        adv_model = FGSM(target_model_dir=target_model_folder)

    elif adv_model_type == 'pgd':
        adv_model = PGD(target_model_dir=target_model_folder)
    else:
        print("This attack is not implemented!")
        return None
    return adv_model


def load_defense(defense_model_folder=None):
    defense_model_path = f'{defense_model_folder}/{Config.APE_GAN_CKPT}'
    defense_model = ApeGan.load_from_checkpoint(defense_model_path, strict=False)
    return defense_model


def evaluate(X, y, attack_model, defense_model=None):
    X_adv = attack_model(X)

    if not check_distance(X, X_adv, eps=args.eps):
        raise Exception("Adversarial samples has too large distance from original samples")

    robust_model = TargetModel()
    robust_model.load_state_dict(torch.load(ROBUST_MODEL_PATH))
    robust_model.eval()

    with torch.no_grad():
        probs = robust_model(X_adv)
        pred = torch.argmax(probs, dim=1)
        adv_accuracy = torch.sum(pred == y) / len(y)
        print(f"Accuracy on adversarial samples is {adv_accuracy.item()}")

        if defense_model is not None:
            X_res = defense_model(X_adv)
            probs = robust_model(X_res)
            pred = torch.argmax(probs, dim=1)
            res_accuracy = torch.sum(pred == y) / len(y)
            print(f"Accuracy on restored samples is {res_accuracy.item()}")
    return adv_accuracy, res_accuracy


def describe_adversarial_model(adv_model, is_whitebox, is_distilled):
    wb = 'whitebox (secret)' if is_whitebox else "blacbkox (adv_trained)"
    dist = 'distilled' if is_distilled else ''
    if adv_model == 'adv_gan':
        return f'AdvGAN with {dist} {wb} target model'
    elif adv_model == 'fgsm':
        return f'FGSM with {wb} target model'
    elif adv_model == 'pgd':
        return f'PGD with {wb} target model'
    else:
        print("This attack is not implemented!")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adv-model", type=str, default='adv_gan',
                        help='Type of the attack model. Supported values: [adv_gan, fgsm, pgd]')
    parser.add_argument("--def-attack-model", type=str, default='adv_gan',
                        help='Type of the model used to train APE-GAN. Supported values: [adv_gan, fgsm, pgd]')
    parser.add_argument("--no-eval-defense", default=False, action='store_true',
                        help='If specified, only attack will be evaluated')

    parser.add_argument("--attack-is-distilled", default=False, action='store_true',
                        help='Specifies that attack uses distilled model. Only matters when adv-model=adv_gan')
    parser.add_argument("--attack-is-blackbox", default=False, action='store_true',
                        help='Specified that attack uses adversarially trained model. Otherwise, it uses secret model')

    parser.add_argument("--def-attack-is-distilled", default=False, action='store_true',
                        help='Specifies that the model attacking APE-GAN uses distilled model. '
                             'Only matters when def-attack-model=adv_gan')
    parser.add_argument("--def-attack-is-blackbox", default=False, action='store_true',
                        help='Specifies that the model attacking APE-GAN uses adversarially trained model.')

    parser.add_argument("--eps", type=float, default=0.3,
                        help='Specifies maximum infinite norm of adversarial perturbations. '
                             'Note that this will only applied to evaluation: '
                             'the model will not be retrained for new epsilon.')
    args = parser.parse_args()

    adv_path_creator = CreatePaths(args.adv_model, args.attack_is_blackbox, args.attack_is_distilled)
    target_model_path, adv_model_folder, _ = adv_path_creator.create_paths()
    attack_model = load_attack(args.adv_model, adv_model_folder=adv_model_folder, target_model_folder=target_model_path)

    eval_defense = not args.no_eval_defense
    if eval_defense:
        def_path_creator = CreatePaths(args.adv_model, args.def_attack_is_blackbox, args.def_attack_is_distilled)
        _, _, defense_model_folder = def_path_creator.create_paths()
        defense_model = load_defense(defense_model_folder=defense_model_folder)
    else:
        defense_model = None

    test_data = datasets.MNIST('mnist', train=False, download=True)
    X = test_data.data
    X = X / 255
    X = torch.unsqueeze(X, 1)
    y = test_data.targets
    adv_accuracy, res_accuracy = evaluate(X, y, attack_model, defense_model)

    attack_desc = describe_adversarial_model(args.adv_model, args.attack_is_blackbox, args.attack_is_distilled)
    print(f"Adversarial examples are generated by {attack_desc}")
    if eval_defense:
        def_desc = describe_adversarial_model(args.adv_model, args.def_attack_is_blackbox, args.def_attack_is_distilled)
        print(f"APE-GAN was trained on {def_desc} \n")

    with open("evaluate_results.txt", "a") as file:
        file.write(f'Attack: {attack_desc}\n')
        file.write(f'{adv_accuracy.item()}\n')

        if eval_defense:
            file.write(f'Restored: {def_desc}\n')
            file.write(f'{res_accuracy.item()}\n')

        file.write(f'\n')
