import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cleverhans.torch.utils import optimize_linear

from models.target_models.target_model import TargetModel

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent


class FGSM(nn.Module):
    def __init__(self,
                 eps=0.3,
                 norm=np.inf,
                 clip_min=0,
                 clip_max=1,
                 target_model_dir='../target_models/pytorch/adv_trained.ckpt',
                 target_model=None):
        super(FGSM, self).__init__()

        self.eps = eps
        self.norm = norm
        self.clip_min = clip_min
        self.clip_max = clip_max

        if target_model is None:
            self.target_model = TargetModel()
            self.target_model.load_state_dict(torch.load(target_model_dir))
            self.target_model.freeze()
            self.target_model.eval()
        else:
            self.target_model = target_model

    def forward(self, imgs):
        # imgs_fgsm = fast_gradient_method(
        #     self.target_model,
        #     imgs,
        #     self.eps,
        #     self.norm,
        #     self.clip_min,
        #     self.clip_max
        # )
        imgs_fgsm = self.fast_gradient_method(imgs)

        return imgs_fgsm

    def fast_gradient_method(
            self,
            x
    ):
        # x = x.clone().detach().to(torch.float).requires_grad_(True)
        # _, y = torch.max(model_fn(x), 1)

        # loss_fn = torch.nn.CrossEntropyLoss()
        # loss = loss_fn(model_fn(x), y)

        # Define gradient of loss wrt input
        # loss.backward()
        grad = self.target_model.gradient(x)
        optimal_perturbation = optimize_linear(grad, self.eps, self.norm)

        # Add perturbation to original example to obtain adversarial example
        adv_x = x + optimal_perturbation

        adv_x = torch.clamp(adv_x, self.clip_min, self.clip_max)
        return adv_x


class PGD(nn.Module):
    def __init__(self,
                 eps=0.3,
                 eps_iter=0.01,
                 nb_iter=40,
                 norm=np.inf,
                 clip_min=0,
                 clip_max=1,
                 target_model_dir='../target_models/pytorch/adv_trained.ckpt',
                 target_model=None):
        super(PGD, self).__init__()

        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.norm = norm
        self.clip_min = clip_min
        self.clip_max = clip_max

        if target_model is None:
            self.target_model = TargetModel()
            self.target_model.load_state_dict(torch.load(target_model_dir))
            self.target_model.freeze()
            self.target_model.eval()
        else:
            self.target_model = target_model

    def forward(self, imgs):
        imgs_pgd = projected_gradient_descent(
            self.target_model,
            imgs,
            self.eps,
            self.eps_iter,
            self.nb_iter,
            self.norm,
            self.clip_min,
            self.clip_max
        )

        return imgs_pgd


class CW_L2(nn.Module):
    def __init__(self,
                 n_classes=10,
                 target_model_dir='../target_models/pytorch/adv_trained.ckpt'):
        super(CW_L2, self).__init__()

        self.n_classes = n_classes

        self.target_model = TargetModel()
        self.target_model.load_state_dict(torch.load(target_model_dir))
        self.target_model.freeze()
        self.target_model.eval()

    def forward(self, imgs):
        imgs_cw_l2 = carlini_wagner_l2(self.target_model, imgs, self.n_classes)

        return imgs_cw_l2
