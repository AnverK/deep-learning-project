import argparse
import os

import numpy as np
import tensorflow.compat.v1 as tf
from torch.utils.data import DataLoader

import torch

from config import Config
from models.adv_gan_lightning.robust_model import Model
from models.adv_gan_lightning.target_model import TargetModel
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor

tf.disable_v2_behavior()


def load_weights(sess, torch_model):
    tf2torch_names = {
        'Variable': 'input_net.0.weight',
        'Variable_1': 'input_net.0.bias',
        'Variable_2': 'input_net.3.weight',
        'Variable_3': 'input_net.3.bias',
        'Variable_4': 'output_net.1.weight',
        'Variable_5': 'output_net.1.bias',
        'Variable_6': 'output_net.3.weight',
        'Variable_7': 'output_net.3.bias',
    }
    state_dict = torch_model.state_dict()
    for var in tf.get_collection('trainable_variables'):
        k = var.name[:-2]
        if k not in tf2torch_names:
            continue
        v = tf2torch_names[k]
        np_tensor = sess.run(var)
        torch_tensor = torch.from_numpy(np_tensor)
        if len(var.shape) == 2:
            torch_tensor = torch_tensor.T
        elif len(var.shape) == 4:
            torch_tensor = torch_tensor.permute(3, 2, 0, 1)
        state_dict[v] = torch_tensor
    print(torch_model.load_state_dict(state_dict))
    return torch_model


def compare_models(tf_model, torch_model):
    torch_model.freeze()
    torch_model.eval()
    dm = mnist.MNIST('mnist', download=False, train=False, transform=ToTensor())
    dl = DataLoader(dm, 10000)
    with torch.no_grad():
        for x, y in dl:
            probs = torch_model(x)
            preds = torch.argmax(probs, dim=1)
            acc_torch = torch.sum(preds == y) / len(y)
            print(f"Pytorch model accuracy: {acc_torch}")

            x = x.data.cpu().numpy().reshape(-1, 28 * 28)
            y = y.data.cpu().numpy()
            probs = tf_model.pre_softmax.eval(session=sess, feed_dict={tf_model.x_input: x})
            preds = np.argmax(probs, axis=1)
            acc_tf = np.sum(preds == y) / len(y)
            print(f"Tensorflow model accuracy: {acc_tf}")
    if np.abs(acc_tf - acc_torch) > 1e-3 or acc_torch < 0.9:
        print("Accuracies are different, some error occurred")
        return False
    else:
        print("OK! Accuracies are the same!")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path to the original weights of the model from mnist_challenge
    parser.add_argument("--tf-checkpoint-path", type=str,
                        default=f"{Config.LOGS_PATH}/{Config.TARGET_MODEL_FOLDER}/natural")
    # path where to save the pytorch model
    parser.add_argument("--torch-checkpoint-path", type=str,
                        default=f"{Config.LOGS_PATH}/{Config.TARGET_MODEL_FOLDER}/converted_natural")
    args = parser.parse_args()

    torch_model = TargetModel()

    sess = tf.Session(config=tf.ConfigProto(
        device_count={'GPU': 0}
    ))
    tf_model = Model()

    model_file = tf.train.latest_checkpoint(args.tf_checkpoint_path)
    saver = tf.train.Saver()
    saver.restore(sess, model_file)

    load_weights(sess, torch_model)
    # torch_model.load_state_dict(torch.load(f"{args.torch_checkpoint_path}/model.ckpt"))
    compare_models(tf_model, torch_model)

    os.makedirs(args.torch_checkpoint_path, exist_ok=True)
    torch.save(torch_model.state_dict(), f"{args.torch_checkpoint_path}/model.ckpt")
