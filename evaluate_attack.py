import argparse

import torch
from torchvision import datasets

from config import Config
from models.adv_gan_lightning.adv_gan import AdvGAN
from models.adv_gan_lightning.target_model import TargetModel
from models.ape_gan_lightning.ape_gan import ApeGan
import attacks


def check_distance(X, X_adv, eps=0.3):
    norm = torch.max(torch.abs(X - X_adv))
    print(norm)
    return norm <= eps + 1e-5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adv-model-path", type=str, default=f'{Config.LOGS_PATH}/adv_gan_adv_whitebox/last.ckpt')
    parser.add_argument("--robust-model-path", type=str,
                        default=f'{Config.LOGS_PATH}/{Config.TARGET_MODEL_FOLDER}/converted_secret/model.ckpt')

    # pass empty string if you only want to evaluate attack model
    parser.add_argument("--defense-model-path", type=str,
                        default=f'{Config.LOGS_PATH}/ape_gan_adv_blackbox/last.ckpt')
    # currently not used TODO
    parser.add_argument("--dataset", type=str, default='mnist')
    parser.add_argument("--eps", type=float, default=0.3)

    args = parser.parse_args()
    # need to pass hyperparameters, since we didn't save them in adv_gan
    adv_model = AdvGAN.load_from_checkpoint(args.adv_model_path, model_num_labels=10, image_nc=1, box_min=0, box_max=1,
                                            tensorflow=False, is_blackbox=False, is_relativistic=False,
                                            target_model_dir=f'{Config.LOGS_PATH}/{Config.TARGET_MODEL_FOLDER}/converted_adv_trained/model.ckpt')
    adv_model.freeze()
    adv_model.eval()

    eval_defense = args.defense_model_path != ''
    if eval_defense:
        defense_model = ApeGan.load_from_checkpoint(args.defense_model_path,
                                                    in_ch=1,
                                                    gen_loss_scale=Config.APE_GAN_gen_loss_scale,
                                                    dis_loss_scale=Config.APE_GAN_dis_loss_scale,
                                                    lr=Config.APE_GAN_lr,
                                                    attack=adv_model,
                                                    target_model_dir=f'{Config.LOGS_PATH}/{Config.TARGET_MODEL_FOLDER}/converted_adv_trained/model.ckpt'
                                                    )

    if args.dataset == 'mnist':
        test_data = datasets.MNIST('mnist', train=False, download=True)
    else:
        raise Exception("Other datasets are not supported yet")

    """
    adv_model = attacks.FGSM(
                target_model_dir=f'{Config.LOGS_PATH}/{Config.TARGET_MODEL_FOLDER}/converted/adv_trained.ckpt')
    # adv_model = attacks.PGD(
                target_model_dir=f'{Config.LOGS_PATH}/{Config.TARGET_MODEL_FOLDER}/converted/adv_trained.ckpt')
    # adv_model.eval()
    """

    X = test_data.data
    X = X / 255
    X = torch.unsqueeze(X, 1)
    X_adv = adv_model(X)
    y = test_data.targets
    if eval_defense:
        X_res = defense_model.generate_restored(X_adv)

    if not check_distance(X, X_adv, eps=args.eps):
        raise Exception("Adversarial samples has too large distance from original samples")

    robust_model = TargetModel()
    robust_model.load_state_dict(torch.load(args.robust_model_path))

    robust_model.eval()
    with torch.no_grad():
        probs = robust_model.forward(X_adv)
        pred = torch.argmax(probs, dim=1)
        accuracy = torch.sum(pred == y) / len(y)
        print(f"Accuracy on adversarial samples is {accuracy.item()}")

        if eval_defense:
            probs = robust_model.forward(X_res)
            pred = torch.argmax(probs, dim=1)
            accuracy = torch.sum(pred == y) / len(y)
            print(f"Accuracy on restored samples is {accuracy.item()}")
