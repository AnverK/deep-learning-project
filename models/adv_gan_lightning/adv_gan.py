from .discriminator import Discriminator
from .generator import Generator
from .target_model import TargetModel
from .student_model import StudentModel
from .robust_model import Model

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from torchmetrics.functional import accuracy
import wandb

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN(LightningModule):
    def __init__(
            self,
            model_num_labels,
            image_nc,
            box_min,
            box_max,
            lr: float = 1e-3,
            b1: float = 0.5,
            b2: float = 0.999,
            num_batches_to_log=1,
            num_samples_to_log=16,
            is_relativistic=True,
            is_blackbox=True,
            tensorflow=True,
            robust_target_model_dir='../../mnist_challenge/models/adv_trained/',
            target_model_dir='last.ckpt'
    ):
        super().__init__()

        self.save_hyperparameters()

        output_nc = image_nc
        self.model_num_labels = model_num_labels
        self.gen_input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max
        self.b2 = b2
        self.b1 = b1
        self.lr = lr
        self.num_batches_to_log = num_batches_to_log
        self.num_samples_to_log = num_samples_to_log
        self.is_relativistic = is_relativistic
        self.is_blackbox = is_blackbox
        self.tensorflow = tensorflow

        # networks
        self.generator = Generator(self.gen_input_nc, image_nc)
        self.discriminator = Discriminator(image_nc)
        self.student_model = StudentModel()

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.student_model.apply(weights_init)

        # self.model = TargetModel.load_from_checkpoint(checkpoint_path="last.ckpt")

        self.target_model = Model()

        if self.tensorflow:
            self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
            model_file = tf.train.latest_checkpoint(robust_target_model_dir)

            saver = tf.train.Saver()
            saver.restore(self.sess, model_file)

        if not self.tensorflow:
            self.target_model = TargetModel()
            self.target_model.load_state_dict(torch.load(target_model_dir))            
            self.target_model.freeze()
            self.target_model.eval()
        else:
            self.target_model = Model()
            self.sess = tf.Session(config=tf.ConfigProto(
                device_count={'GPU': 0}
            ))
            model_file = tf.train.latest_checkpoint(robust_target_model_dir)

            saver = tf.train.Saver()
            saver.restore(self.sess, model_file)

        # Temperature and Scaling of Losses for the distillation
        self.temp = 1
        self.alpha = 0
        # Used for the perturbation/hinge loss
        self.C = 0.1
        # To scale the importance of losses
        self.gen_lambda = 1
        self.adv_lambda = 150
        self.pert_lambda = 5

        self.epoch_decay = 500

        self.dist_fake_scale = 1
        self.dist_real_scale = 1
        self.dist_tradeoff_epochs = 10
        self.dist_cutoff = 0.7

    def forward(self, z):
        perturbations, adv_imgs = self.generate_adv_imgs(z)

        return adv_imgs

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch

        perturbation, adv_imgs = self.generate_adv_imgs(imgs)

        if optimizer_idx == 0:
            losses = self.generator_losses(imgs, labels, adv_imgs, perturbation, 'train')
            
            return losses["train_loss_generator"]

        if optimizer_idx == 1:
            losses = self.discriminator_loss(imgs, adv_imgs, 'train')

            return losses["train_loss_discriminator"]

        if optimizer_idx == 2:
            losses = self.distillation_loss(imgs, labels, adv_imgs, 'train')

            return losses["train_loss_distillation_total"]

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch

        perturbation, adv_imgs = self.generate_adv_imgs(imgs)
        losses = self.generator_losses(imgs, labels, adv_imgs, perturbation, 'validation')

        labels_original_pred, labels_adversarial_pred = self.target_model_metrics(imgs, labels, adv_imgs)

        return imgs, labels, perturbation, adv_imgs, labels_original_pred, labels_adversarial_pred

    def validation_epoch_end(self, outputs):
        imgs_batches, labels_batches, perturbation_batches, adv_imgs_batches, labels_original_pred_batches, labels_adversarial_pred_batches = [
            torch.stack([output[i] for output in outputs])[:self.num_batches_to_log, :self.num_samples_to_log] for i in
            range(len(outputs[0]))]

        wandb.log({
            "pred_imgs": [
                wandb.Image(
                    img,
                    caption=f'Pred: {pred}, Label: {label}'
                ) for imgs, labels, preds in zip(imgs_batches, labels_batches, labels_original_pred_batches) for
                img, label, pred in zip(imgs, labels, preds)
            ] if self.current_epoch == 0 else None,
            "pred_adv_imgs": [
                wandb.Image(
                    adv_img,
                    caption=f'Pred: {pred}, Label: {label}'
                ) for adv_imgs, labels, preds in zip(adv_imgs_batches, labels_batches, labels_adversarial_pred_batches)
                for adv_img, label, pred in zip(adv_imgs, labels, preds)
            ],
            "perturbation": [
                wandb.Image(
                    perturbation,
                    caption=f'Label: {label}'
                ) for labels, perturbations in zip(labels_batches, perturbation_batches) for label, perturbation in
                zip(labels, perturbations)
            ]
        })

    def target_model_predict(self, imgs):
        if self.tensorflow:
            np_tensor = imgs.data.cpu().numpy()
            np_tensor = np_tensor.reshape(np_tensor.shape[0], -1)

            logits = self.target_model.pre_softmax.eval(session=self.sess,
                                                        feed_dict={self.target_model.x_input: np_tensor})
            return torch.from_numpy(logits)

        return self.target_model(imgs)

    def target_model_metrics(self, imgs, labels, adv_imgs, stage='validation'):
        labels_original_pred = self.target_model_predict(imgs).to(self.device).argmax(1)
        labels_adversarial_pred = self.target_model_predict(adv_imgs).to(self.device).argmax(1)

        accuracy_original = accuracy(labels_original_pred, labels)
        accuracy_adversarial = accuracy(labels_adversarial_pred, labels)

        if self.is_blackbox:
            labels_original_pred_student = self.student_model(imgs).to(self.device).argmax(1)
            labels_adversarial_pred_student = self.student_model(adv_imgs).to(self.device).argmax(1)

            accuracy_original_student = accuracy(labels_original_pred_student, labels)
            accuracy_adversarial_student = accuracy(labels_adversarial_pred_student, labels)

        losses = {
            f"{stage}_accuracy_original": accuracy_original,
            f"{stage}_accuracy_adversarial": accuracy_adversarial,
            f"{stage}_accuracy_original_student": accuracy_original_student if self.is_blackbox else None,
            f"{stage}_accuracy_adversarial_student": accuracy_adversarial_student if self.is_blackbox else None,
        }

        self.log_dict(
            losses,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )

        return labels_original_pred, labels_adversarial_pred

    def generate_adv_imgs(self, imgs):
        perturbation = self.generator(imgs)

        adv_imgs = torch.clamp(perturbation, -0.3, 0.3) + imgs
        adv_imgs = torch.clamp(adv_imgs, self.box_min, self.box_max)

        return perturbation, adv_imgs

    def generator_losses(self, imgs, labels, adv_imgs, perturbation, stage='train'):
        if self.is_relativistic:
            loss_generator = self.generator_loss_relativistic(imgs, adv_imgs)
        else:
            loss_generator = self.generator_loss_fake(adv_imgs)

        loss_perturb = self.perturbation_loss(perturbation)
        loss_adv = self.adversarial_loss(adv_imgs, labels)

        sum_loss_generator = (self.adv_lambda * loss_adv) + (self.gen_lambda * loss_generator) + (
                self.pert_lambda * loss_perturb)

        losses = {
            f"{stage}_loss_generator_fake": loss_generator,
            f"{stage}_loss_perturb": loss_perturb,
            f"{stage}_loss_adv": loss_adv,
            f"{stage}_loss_generator": sum_loss_generator,
        }

        self.log_dict(
            losses,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )

        return losses

    def generator_loss_relativistic(self, imgs, adv_imgs):
        logits_real, pred_real = self.discriminator(imgs)
        logits_fake, _ = self.discriminator(adv_imgs)

        real = torch.ones_like(pred_real, device=self.device)

        lossG_real = torch.mean((logits_real - torch.mean(logits_fake) + real) ** 2)
        lossG_fake = torch.mean((logits_fake - torch.mean(logits_real) - real) ** 2)

        lossG = (lossG_real + lossG_fake) / 2

        return lossG

    def generator_loss_fake(self, adv_imgs):
        _, pred_fake = self.discriminator(adv_imgs)
        valid = torch.ones_like(pred_fake, device=self.device)
        lossG_fake = F.mse_loss(pred_fake, valid)

        return lossG_fake

    # Soft hinge loss to bound the magnitude of the perturbation
    def perturbation_loss(self, perturbation):
        perturbation_norm = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
        loss_hinge = torch.max(torch.zeros(1, device=self.device), perturbation_norm - self.C)

        return loss_hinge
        

        return F.mse_loss(perturbation, torch.zeros_like(perturbation, device=self.device))

    def adversarial_loss(self, adv_imgs, labels):
        # Loss of fooling the target model C&W loss function:
        if self.is_blackbox:
            preds = self.student_model(adv_imgs)
        else:
            preds = self.target_model_predict(adv_imgs).to(self.device)

        probs = F.softmax(preds, dim=1)
        onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

        # Probabilities of ground truth
        real = onehot_labels * probs
        real = torch.sum(real, dim=1)

        # Probabilities of the remaining classes
        # other, _ = torch.max((1 - onehot_labels) * probs - onehot_labels * 10000, dim=1)
        other = (1 - onehot_labels) * probs - onehot_labels
        other, _ = torch.max(other, dim=1)

        zeros = torch.zeros_like(other)

        loss_adv = torch.max(real - other, zeros)
        loss_adv = torch.mean(loss_adv)

        return loss_adv

    def discriminator_loss_real_fake(self, imgs, adv_imgs):
        logits_real, pred_real = self.discriminator(imgs)
        logits_fake, pred_fake = self.discriminator(adv_imgs)

        real = torch.ones_like(pred_real, device=self.device)
        fake = torch.zeros_like(pred_fake, device=self.device)

        if self.is_relativistic:
            lossD_real = torch.mean((logits_real - torch.mean(logits_fake) - real) ** 2)
            lossD_fake = torch.mean((logits_fake - torch.mean(logits_real) + real) ** 2)
        else:
            lossD_real = F.mse_loss(pred_real, real)
            lossD_fake = F.mse_loss(pred_fake, fake)

        return lossD_real, lossD_fake

    def discriminator_loss(self, imgs, adv_imgs, stage='train'):
        loss_real, loss_fake = self.discriminator_loss_real_fake(imgs, adv_imgs)

        if self.is_relativistic:
            loss_discriminator = (loss_real + loss_fake) / 2
        else:
            loss_discriminator = loss_real + loss_fake

        losses = {
            f"{stage}_loss_discriminator": loss_discriminator,
        }

        self.log_dict(
            losses,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )

        return losses

    def distillation_loss(self, imgs, labels, adv_imgs, stage='train'):
        teacher_preds_real = self.target_model_predict(imgs).to(self.device)
        teacher_preds_fake = self.target_model_predict(adv_imgs).to(self.device)

        student_preds_real = self.student_model(imgs)
        student_preds_fake = self.student_model(adv_imgs)

        student_loss = F.cross_entropy(student_preds_real, labels)

        loss_dist_real = F.cross_entropy(
            student_preds_real,
            torch.argmax(F.softmax(teacher_preds_real / self.temp, dim=1), dim=1),
        )

        loss_dist_fake = F.cross_entropy(
            student_preds_fake,
            torch.argmax(F.softmax(teacher_preds_fake / self.temp, dim=1), dim=1),
        )

        loss_distillation = self.dist_real_scale * loss_dist_real + self.dist_fake_scale * loss_dist_fake
        loss_distillation_total = self.alpha * student_loss + (1 - self.alpha) * loss_distillation

        losses = {
            f"{stage}_loss_distillation_total": loss_distillation_total,
            f"{stage}_student_loss": student_loss,
            f"{stage}_loss_distillation": loss_distillation,
            f"{stage}_loss_distillation_real": loss_dist_real,
            f"{stage}_loss_distillation_fake": loss_dist_fake,
        }

        self.log_dict(
            losses,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )

        return losses

    def optimizer_step(
            self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            optimizer_closure,
            on_tpu=False,
            using_native_amp=False,
            using_lbfgs=False,
    ):
        if optimizer_idx == 0:
            optimizer.step(closure=optimizer_closure)
            optimizer.step(closure=optimizer_closure)

        if optimizer_idx == 1:
            optimizer.step(closure=optimizer_closure)

        if self.is_blackbox and optimizer_idx == 2:
            optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))


        if not self.is_blackbox:
            return [opt_g, opt_d], []

        opt_sm = torch.optim.Adam(self.student_model.parameters(), lr=1e-4)
        

        return [opt_g, opt_d, opt_sm], []
