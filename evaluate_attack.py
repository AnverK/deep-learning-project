import argparse

import torch
from torchvision import datasets

from config import Config
from models.adv_gan_lightning.adv_gan import AdvGAN
from models.adv_gan_lightning.target_model import TargetModel


def check_distance(X, X_adv, eps=0.3):
    norm = torch.max(torch.abs(X - X_adv))
    print(norm)
    return norm <= eps + 1e-5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adv-model-path", type=str, default=f'{Config.LOGS_PATH}/adv_gan_adv_whitebox/last.ckpt')
    parser.add_argument("--robust-model-path", type=str,
                        default=f'{Config.LOGS_PATH}/{Config.TARGET_MODEL_FOLDER}/converted_secret/model.ckpt')
    # currently not used TODO
    parser.add_argument("--dataset", type=str, default='mnist')
    parser.add_argument("--eps", type=float, default=0.3)

    args = parser.parse_args()
    # need to pass hyperparameters, since we didn't save them in adv_gan
    adv_model = AdvGAN.load_from_checkpoint(args.adv_model_path, model_num_labels=10, image_nc=1, box_min=0, box_max=1,
                                            tensorflow=False, is_blackbox=False, is_relativistic=False,
                                            target_model_dir=f'{Config.LOGS_PATH}/{Config.TARGET_MODEL_FOLDER}/converted_adv_trained/model.ckpt')

    if args.dataset == 'mnist':
        test_data = datasets.MNIST('mnist', train=False, download=True)
    else:
        raise Exception("Other datasets are not supported yet")

    X = test_data.data
    X = X / 255
    X = torch.unsqueeze(X, 1)
    _, X_adv = adv_model.generate_adv_imgs(X)
    y = test_data.targets

    if not check_distance(X, X_adv, eps=args.eps):
        raise Exception("Adversarial samples has too large distance from original samples")

    robust_model = TargetModel()
    robust_model.load_state_dict(torch.load(args.robust_model_path))

    robust_model.eval()
    with torch.no_grad():
        probs = robust_model.forward(X_adv)
        pred = torch.argmax(probs, dim=1)
        accuracy = torch.sum(pred == y) / len(y)
        print(f"Accuracy is {accuracy.item()}")
