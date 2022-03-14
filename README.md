# Deep-Learning-Project

Group project for the course Deep Learning taught in the autumn semester 2021 at ETH

## Summary

The goal of this project is to research different adversarial attacks and defences. Moreover, we suggest a new type of defence (adv_gan_reverse) and re-evaluate [MNIST Challenge](https://github.com/MadryLab/mnist_challenge). Shortly, MNIST Challenge is focused on attacking some secret, robust model. However, we apply [APE-GAN](https://arxiv.org/abs/1707.05474), which should eliminate some of the adversarial perturbations. Here is an example:

<p float="left" align="middle">
  <img src="/wandb_images/orignal.png" width="32%" />
  <img src="/wandb_images/adversarial.png" width="32%" /> 
  <img src="/wandb_images/restored.png" width="32%" />
</p>

The left image is original and is correctly classified as 5 by a robust model. In the middle, it's the same image but after a successful adversarial attack by [AdvGAN](https://arxiv.org/abs/1801.02610) (classified as 6). And the right image is the same from the middle after eliminating the adversarial perturbation by APE-GAN. And it's classified correctly again!

For a more detailed explanation and more examples, please read our [paper](https://github.com/AnverK/deep-learning-project/blob/main/GAN%20Wars%20(paper).pdf).

Last but not least, we tried to use PyTorch lightning modules as much as possible, so it might be convenient to re-use them. We also added [WandB logging](#wandb-logging), which makes it much easier to analyse the results and training.

## How to set up the repository

1. Clone this repository and also challenge repository: https://github.com/MadryLab/mnist_challenge:

```
git clone https://gitlab.ethz.ch/osaeedi/deep-learning-project/
cd deep-learning-project
git clone https://github.com/MadryLab/mnist_challenge
```

2. Download checkpoints of the model given by the challenge authors on Tensorflow:

```
cd mnist_challenge
python fetch_model.py adv_trained
python fetch_model.py secret
```

3. Convert each of these checkpoints to Pytorch checkpoints:

```
cd ..
python converter.py --tf-checkpoint-path=mnist_challenge/models/adv_trained --torch-checkpoint-path=dl_logs/target_model/converted_adv_trained
python converter.py --tf-checkpoint-path=mnist_challenge/models/secret --torch-checkpoint-path=dl_logs/target_model/converted_secret
```

If everything was correct, you should see the message: `OK! Accuracies are the same!` when run each of the converters.
After that, you can remove mnist_challenge repository:

```
rm -r mnist_challenge
```

4. \[OPTIONAL\] If you want to customize the paths (e.g., you want to store checkpoints and other logs in other folder),
   you should change the `config.py` correspondingly:
    * `LOGS_PATH` — path to the root folder which contains all the data: checkpoints, wandb logs, dataset...
    * `TARGET_MODEL_FOLDER` — name of the folder with Pytorch checkpoints of target model (specified in the previous
      step)
    * `TARGET_MODEL_WHITE_BOX_FOLDER` — name of the folder with checkpoint of the secret model (should be name of the
      folder inside `TARGET_MODEL_FOLDER`)
    * `TARGET_MODEL_BLACK_BOX_FOLDER` — name of the folder with checkpoint of the adversarially trained model (should be
      name of the folder inside `TARGET_MODEL_FOLDER`)

For example, the secret model should be accessed by `{LOGS_PATH}/{TARGET_MODEL_FOLDER}/{TARGET_MODEL_WHITE_BOX_FOLDER}`

## How to train AdvGAN

Train AdvGAN model running

```
python train_adv_gan.py [--is-blackbox] [--is-distilled] 
```

Flag `--is-blackbox` points that AdvGAN would be trained on adversarially trained target model. Otherwise, it will be
trained on secret target model (the same as for evaluation).

Flag `--is-distilled` points that AdvGAN would be trained on "student" model which is trained on the specified target
model. So the target model is more like an oracle (or teacher) which gives the labels, but AdvGAN doesn't have access to
the target model itself.

Example of the path to corresponding checkpoint is `{LOGS_PATH}/{ADV_GAN_FOLDER}/blackbox/not_distilled`
It will take few hours on CPU, so we highly recommend running it on GPU. Sometimes, when running on GPU, there is a
problem with wandb logging. Running this command should help:

```
module load eth_proxy
```

## How to train APE-GAN

Train APE-GAN model running

```
python train_ape_gan.py [--adv-model=adv_gan] [--is-blackbox] [--is-distilled] 
```

Parameter `--adv-model` corresponds to the attack model, from which APE-GAN tries to defend. Currently supported values
here are adv_gan, fgsm and pgd. We use implementation of FGSM with CE loss. PGD uses 40 iterations and is quite slow

Flag `--is-blackbox` points at type of the target model.

Flag `--is-distilled` only matters for `--adv-model=adv_gan`

Example of the path to corresponding checkpoint is `{LOGS_PATH}/{APE_GAN_FOLDER}/adv_gan_whitebox_not_distilled`. Note
that name of the folder corresponds to the attack model. Also, if you ran with `adv-model=adv_gan`, make sure that such
version of AdvGAN (with specified blackbox and distilled flags) was trained on the previous step. This model we also
highly recommend training on GPU.

## How to evaluate the results

```
python evaluate_attack.py 
```

It has many parameters, so we added help for this script, however it's really easy to use. Example:

```
python evaluate_attack.py --adv-model=adv_gan --attack-is-blackbox --def-attack-model=fgsm --def-attack-is-distilled
```

This script evaluates performance of AdvGAN which was trained using adversarially trained model (not a secret one). It
also evaluates performance of APE-GAN against this AdvGAN attack while it was trained on fgsm attack with a secret model
as target. The last flag (`--def-attack-is-distilled`) is ignored since distilled model is only supported for AdvGAN in
our implementation.

## WandB logging

To look at the charts or at the images, we use [wandb](https://wandb.ai/). We highly recommend using it because it
provides real-time plots and images (adversarial and restored) during the training. Simply sign up there, login in the
first running and access it! If you don't want to use this feature, pass `--no-wandb` to `train_[adv|ape]_gan.py`.
