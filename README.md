# Deep-Learning-Project

Group project for the course Deep Learning taught in the autumn semester 2021 at ETH

## How to reproduce the results

1. Clone this repository and also challenge repository: https://github.com/MadryLab/mnist_challenge
2. In the challenge repository, run
 ```
 python fetch_model.py [natural|adv_trained|secret]
 ```
to download the checkpoints of the model given by the challenge authors on Tensorflow 
3. Convert these checkpoints to Pytorch checkpoints running:
 ```
 python converter.py --tf-checkpoint-path=path/to/tf-checkpoints --torch-checkpoint-path=path/to/save/torch-checkpoints
 ```
4. Train AdvGAN model running 