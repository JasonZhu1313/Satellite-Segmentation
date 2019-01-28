from Model import Model
from Config import Config

from util import _variable_with_weight_decay, _variable_on_cpu
from customer_init import orthogonal_initializer
import tensorflow as tf

import math
class SegnetModel(Model):
    def __init__(self):
        self.config = Config()

    def add_placeholders(self):
        self.train_data_node = tf.placeholder(tf.float32, shape=[self.config.BATCH_SIZE,
            self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, self.config.IMAGE_DEPTH])
        self.train_label_node = tf.placeholder(tf.int8, shape=[self.config.BATCH_SIZE, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH,1])
        self.phase_train = tf.placeholder(tf.bool, name="phase_train")

        self.average_pl = tf.placeholder(tf.float32)
        self.acc_pl = tf.placeholder(tf.float32)
        self.iu_pl = tf.placeholder(tf.float32)

        self.test_data_node = tf.placeholder(
            tf.float32,
            shape=[self.config.BATCH_SIZE,
            self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, self.config.IMAGE_DEPTH])

        self.test_labels_node = tf.placeholder(tf.int64, shape=[self.config.BATCH_SIZE, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH,1])

    def create_feed_dict(self):
        pass

    def add_prediction_op(self):
        pass

    def add_loss_op(self, pred):
        pass

    def add_training_op(self, loss):
        pass

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        pass

    def predict_on_batch(self, sess, inputs_batch):
        # norm1
        norm1 = tf.nn.lrn(self.train_data_node, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
                          name='norm1')

        # conv1
        conv1 = self.conv_layer_with_bn(norm1, [7, 7, self.train_data_node.get_shape().as_list()[3], 64], self.phase_train, name="conv1")

        # pool1
        pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                          padding='SAME', name='pool1')

        # conv2
        conv2 = self.conv_layer_with_bn(pool1, [7, 7, 64, 64], self.phase_train, name="conv2")

        # pool2
        pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1],
                                                          strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        # conv3
        conv3 = self.conv_layer_with_bn(pool2, [7, 7, 64, 64], self.phase_train, name="conv3")

        # pool3
        pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3, ksize=[1, 2, 2, 1],
                                                          strides=[1, 2, 2, 1], padding='SAME', name='pool3')
        # conv4
        conv4 = self.conv_layer_with_bn(pool3, [7, 7, 64, 64], self.phase_train, name="conv4")

        """ End of encoder """
        """ start upsample """


        # pool4
        pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1],
                                                          strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def conv_layer_with_bn(self, inputT, shape, train_phase, activation = True, name = None):
        in_channel = shape[2]
        out_channel = shape[3]
        k_size = shape[0]
        with tf.variable_scope(name) as scope:
            kernel = _variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
            conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            if activation is True:
                conv_out = tf.nn.relu(self.batch_norm_layer(bias, train_phase, scope.name))
            else:
                conv_out = self.batch_norm_layer(bias, train_phase, scope.name)
        return conv_out

    def batch_norm_layer(self, inputT, is_training, scope):
        return tf.cond(is_training,
                       lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                                                            center=False, updates_collections=None,
                                                            scope=scope + "_bn"),
                       lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                                                            updates_collections=None, center=False, scope=scope + "_bn",
                                                            reuse=True))


