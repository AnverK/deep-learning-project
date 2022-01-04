# Deep-Learning-Project

Group project for the course Deep Learning taught in the autumn semester 2021 at ETH

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
python converter.py --tf-checkpoint-path=mnist_challenge/models/adv_trained --torch-checkpoint-path=dl_logs/target_model/adv_trained
python converter.py --tf-checkpoint-path=mnist_challenge/models/secret --torch-checkpoint-path=dl_logs/target_model/secret
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
It will take few hours on CPU, so we highly recommend running it on GPU

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

