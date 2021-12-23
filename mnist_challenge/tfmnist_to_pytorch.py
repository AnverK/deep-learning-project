import tensorflow as tf
import tensorflow_datasets
import os
import torch

scratch_path = '/cluster/scratch/mboss'

mnist = tensorflow_datasets.load(
    'mnist', 
    data_dir=f'{scratch_path}/dl_logs/mnist_tensorflow',
    as_supervised=True
)

dir_path = f'{scratch_path}/dl_logs/mnist'

os.makedirs(dir_path, exist_ok=True)

os.makedirs(f'{dir_path}/train', exist_ok=True)
for i, image_label in enumerate(mnist['train']):
    image, label = image_label
    
    os.makedirs(f'{dir_path}/train/{label}', exist_ok=True)
    
    image_torch = torch.from_numpy(image.numpy().transpose(2, 0, 1))
    torch.save(image_torch, f'{dir_path}/train/{label}/{i}.pt')

os.makedirs(f'{dir_path}/test', exist_ok=True)
for i, image_label in enumerate(mnist['test']):
    image, label = image_label

    os.makedirs(f'{dir_path}/test/{label}', exist_ok=True)
    
    image_torch = torch.from_numpy(image.numpy().transpose(2, 0, 1))
    torch.save(image_torch, f'{dir_path}/test/{label}/{i}.pt')