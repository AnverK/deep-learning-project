<<<<<<< HEAD
import argparse

import numpy as np
import torch
from torchvision import datasets
import tensorflow.compat.v1 as tf

from adv_gan_lightning.robust_model import Model

tf.disable_v2_behavior()


def check_distance(X, X_adv, eps=0.3):
    norm = torch.norm(X - X_adv, float('inf'), dim=1)
    small_distance = norm <= eps
    return torch.all(small_distance)


def target_model_logits(X):
    np_tensor = X.data.cpu().numpy()

    logits = robust_model.pre_softmax.eval(session=sess, feed_dict={robust_model.x_input: np_tensor})
    return torch.from_numpy(logits)


def target_model_preds(X):
    logits = target_model_logits(X)
    return torch.argmax(logits, dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # numpy array with with values in [0, 1] and flattened images (i.e., it's 2D array)
    parser.add_argument("--adv-samples-path", type=str, default="./adv_gan_attack.npy")
    parser.add_argument("--model-path", type=str, default="models/adv_trained")
    # currently not used TODO
    parser.add_argument("--dataset", type=str, default='mnist')
    parser.add_argument("--eps", type=float, default=0.3)

    args = parser.parse_args()

    X_adv = np.load(args.adv_samples_path)
    X_adv = torch.from_numpy(X_adv)

    if args.dataset == 'mnist':
        test_data = datasets.MNIST('mnist', train=False, download=True)
    else:
        raise Exception("Other datasets are not supported yet")
    X = test_data.data
    y = test_data.targets

    X = X.resize(X.shape[0], X.shape[1] * X.shape[2]) / 255

    if not check_distance(X, X_adv, eps=args.eps):
        raise Exception("Adversarial samples has too large distance from original samples")

    sess = tf.Session()
    robust_model = Model()
    model_file = tf.train.latest_checkpoint('models/adv_trained')

    saver = tf.train.Saver()
    saver.restore(sess, model_file)

    adv_preds = target_model_preds(X_adv)
    correct = torch.sum(y == adv_preds)
    accuracy = correct / len(y)
    print(f"Accuracy is {accuracy}")
    exit(0)
=======
import argparse

import numpy as np
import torch
from torchvision import datasets
import tensorflow.compat.v1 as tf

from adv_gan_lightning.robust_model import Model

tf.disable_v2_behavior()


def check_distance(X, X_adv, eps=0.3):
    norm = torch.norm(X - X_adv, float('inf'), dim=1)
    small_distance = norm <= eps
    return torch.all(small_distance)


def target_model_logits(X):
    np_tensor = X.data.cpu().numpy()

    logits = robust_model.pre_softmax.eval(session=sess, feed_dict={robust_model.x_input: np_tensor})
    return torch.from_numpy(logits)


def target_model_preds(X):
    logits = target_model_logits(X)
    return torch.argmax(logits, dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # numpy array with with values in [0, 1] and flattened images (i.e., it's 2D array)
    parser.add_argument("--adv-samples-path", type=str, default="./adv_gan_attack.npy")
    parser.add_argument("--model-path", type=str, default="models/adv_trained")
    # currently not used TODO
    parser.add_argument("--dataset", type=str, default='mnist')
    parser.add_argument("--eps", type=float, default=0.3)

    args = parser.parse_args()

    X_adv = np.load(args.adv_samples_path)
    X_adv = torch.from_numpy(X_adv)

    if args.dataset == 'mnist':
        test_data = datasets.MNIST('mnist', train=False, download=True)
    else:
        raise Exception("Other datasets are not supported yet")
    X = test_data.data
    y = test_data.targets

    X = X.resize(X.shape[0], X.shape[1] * X.shape[2]) / 255

    if not check_distance(X, X_adv, eps=args.eps):
        raise Exception("Adversarial samples has too large distance from original samples")

    sess = tf.Session()
    robust_model = Model()
    model_file = tf.train.latest_checkpoint('models/adv_trained')

    saver = tf.train.Saver()
    saver.restore(sess, model_file)

    adv_preds = target_model_preds(X_adv)
    correct = torch.sum(y == adv_preds)
    accuracy = correct / len(y)
    print(f"Accuracy is {accuracy}")
    exit(0)
>>>>>>> new-master
