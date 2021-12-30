from models.adv_gan_lightning.target_model import TargetModel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)


class FGSM(nn.Module):
    def __init__(self,
                 eps=0.3,
                 norm=np.inf,
                 target_model_dir='../target_models/pytorch/adv_trained.ckpt'):
        super(FGSM, self).__init__()

        self.eps = eps
        self.norm = norm

        self.target_model = TargetModel()
        self.target_model.load_state_dict(torch.load(target_model_dir))
        self.target_model.freeze()
        self.target_model.eval()

    def forward(self, imgs):
        imgs_fgsm = fast_gradient_method(self.target_model, imgs, self.eps, self.norm)

        return imgs_fgsm


class PGD(nn.Module):
    def __init__(self,
                 eps=0.3,
                 eps_iter=0.01,
                 nb_iter=40,
                 norm=np.inf,
                 target_model_dir='../target_models/pytorch/adv_trained.ckpt'):
        super(PGD, self).__init__()

        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.norm = norm

        self.target_model = TargetModel()
        self.target_model.load_state_dict(torch.load(target_model_dir))
        self.target_model.freeze()
        self.target_model.eval()

    def forward(self, imgs):
        imgs_pgd = projected_gradient_descent(self.target_model, imgs, self.eps, self.eps_iter, self.nb_iter, self.norm)

        return imgs_pgd
