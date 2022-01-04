import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

import wandb
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class TargetModel(LightningModule):

    def forward(self, x):
        x = self.input_net(x)
        x = x.permute(0, 2, 3, 1)  # CRUCIAL MAGIC FOR TF-compatibility
        x = self.output_net(x)
        return x

    def configure_optimizers(self):
        lr = self.hparams.lr

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def training_step(self, train_batch, batch_idx):
        preds, loss = self.shared_step(train_batch)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        preds, loss = self.shared_step(val_batch)

        self.log('val_loss', loss)

        return val_batch, preds

    def validation_epoch_end(self, outputs):
        batches = [output[0] for output in outputs]
        imgs_batches = torch.stack([batch[0] for batch in batches])[:self.num_batches_to_log, :self.num_samples_to_log]
        labels_batches = torch.stack([batch[1] for batch in batches])[:self.num_batches_to_log,
                         :self.num_samples_to_log]
        preds_batches = torch.stack([output[1] for output in outputs])[:self.num_batches_to_log,
                        :self.num_samples_to_log].argmax(-1)

        wandb.log({
            "pred": [
                wandb.Image(
                    img,
                    caption=f'Pred: {pred}, Label: {label}'
                ) for imgs, labels, preds in zip(imgs_batches, labels_batches, preds_batches) for img, pred, label in
                zip(imgs, labels, preds)
            ]
        })


class TargetModelMNIST(TargetModel):
    def __init__(
            self,
            lr: float = 0.001,
            num_samples_to_log=16,
            num_batches_to_log=1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.output_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

        self.num_samples_to_log = num_samples_to_log
        self.num_batches_to_log = num_batches_to_log

    def shared_step(self, batch):
        imgs, labels = batch

        preds = torch.zeros(labels.size(0), 10, device=self.device)
        preds = self.input_net(imgs)
        preds = preds.view(-1, 64 * 7 * 7)
        preds = self.output_net(preds)

        loss = self.loss(preds, labels)

        return preds, loss


class ModelCIFAR10(object):
    """ResNet model."""

    def __init__(self, mode):
        """ResNet constructor.

        Args:
          mode: One of 'train' and 'eval'.
        """
        self.mode = mode
        self._build_model()

    def add_internal_summaries(self):
        pass

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model(self):
        assert self.mode == 'train' or self.mode == 'eval'
        """Build the core model within the graph."""
        with tf.variable_scope('input'):

            self.x_input = tf.placeholder(
                tf.float32,
                shape=[None, 32, 32, 3])

            self.y_input = tf.placeholder(tf.int64, shape=None)

            input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                                           self.x_input)
            x = self._conv('init_conv', input_standardized, 3, 3, 16, self._stride_arr(1))

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        res_func = self._residual

        # wide residual network (https://arxiv.org/abs/1605.07146v1)
        # use filters = [16, 16, 32, 64] for a non-wide version
        filters = [16, 160, 320, 640]

        # Update hps.num_residual_units to 9

        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                         activate_before_residual[0])
        for i in range(1, 5):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                         activate_before_residual[1])
        for i in range(1, 5):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                         activate_before_residual[2])
        for i in range(1, 5):
            with tf.variable_scope('unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, 0.1)
            x = self._global_avg_pool(x)

        with tf.variable_scope('logit'):
            self.pre_softmax = self._fully_connected(x, 10)

        self.predictions = tf.argmax(self.pre_softmax, 1)
        self.correct_prediction = tf.equal(self.predictions, self.y_input)
        self.num_correct = tf.reduce_sum(
            tf.cast(self.correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(
            tf.cast(self.correct_prediction, tf.float32))

        self.labels = tf.argmax(self.pre_softmax, axis=1)
        with tf.variable_scope('costs'):
            self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.pre_softmax, labels=self.labels)
            self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
            self.mean_xent = tf.reduce_mean(self.y_xent)
            self.weight_decay_loss = self._decay()

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.name_scope(name):
            return tf.layers.batch_normalization(
                inputs=x,
                momentum=1 - .9,
                center=True,
                scale=True,
                name='BatchNorm')

    def _residual(self, x, in_filter, out_filter, stride,
                  activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, 0.1)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find('DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        return tf.add_n(costs)

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        num_non_batch_dimensions = len(x.shape)
        prod_non_batch_dimensions = 1
        for ii in range(num_non_batch_dimensions - 1):
            prod_non_batch_dimensions *= int(x.shape[ii + 1])
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        w = tf.get_variable(
            'DW', [prod_non_batch_dimensions, out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])


class TargetModelCIFAR10(TargetModel):
    def __init__(
            self,
            tf_model,
            sess
    ):
        super().__init__()
        self.tf_model = tf_model
        self.sess = sess

    def forward(self, X):
        # we have (B, C, H, W), want (B, H, W, C)
        X = X.permute(0, 2, 3, 1)
        X = X.detach().cpu().numpy()
        probs = self.tf_model.pre_softmax.eval(session=self.sess, feed_dict={self.tf_model.x_input: X})

        probs = torch.from_numpy(probs)
        return probs

    def gradient(self, X):
        # we have (B, C, H, W), want (B, H, W, C)
        X = X.permute(0, 2, 3, 1)
        X = X.detach().cpu().numpy()
        grad = tf.gradients(self.tf_model.y_xent, self.tf_model.x_input)
        grad = self.sess.run(grad, feed_dict={self.tf_model.x_input: X})
        grad = torch.from_numpy(grad[0])
        grad = grad.permute(0, 3, 1, 2)
        return grad.to(self.device)
