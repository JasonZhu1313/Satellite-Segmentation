from Model import Model
from Config import Config
from math import ceil
import readfile
import customer_init
import numpy as np
import time
import datetime
import util
import os
import random
from tempfile import TemporaryFile
from customer_init import orthogonal_initializer
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

import math
class SegnetModel(Model):
    def __init__(self):
        self.config = Config()

    def add_placeholders(self):
        self.train_data_node = tf.placeholder(tf.float32, shape=[self.config.BATCH_SIZE,
            self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, self.config.IMAGE_DEPTH])
        self.train_label_node = tf.placeholder(tf.int32, shape=[self.config.BATCH_SIZE, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH,1])
        self.phase_train = tf.placeholder(tf.bool, name="phase_train")

        self.average_pl = tf.placeholder(tf.float32)
        self.acc_pl = tf.placeholder(tf.float32)
        self.iu_pl = tf.placeholder(tf.float32)


        self.test_data_node = tf.placeholder(
            tf.float32,
            shape=[self.config.TEST_BATCH_SIZE,
            self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, self.config.IMAGE_DEPTH])

        self.test_labels_node = tf.placeholder(tf.int64, shape=[self.config.TEST_BATCH_SIZE, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH,1])


    def add_loss_op(self, pred):
        pass

    def add_training_op(self, total_loss):
        """ fix lr """
        lr = self.config.INITIAL_LEARNING_RATE
        loss_averages_op = util._add_loss_summaries(total_loss)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(lr)
            grads = opt.compute_gradients(total_loss)
        apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            self.config.MOVING_AVERAGE_DECAY, self.global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        pass

    def add_prediction_op(self):
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
        # upsample4
        # Need to change when using different dataset out_w, out_h
        # upsample4 = upsample_with_pool_indices(pool4, pool4_indices, pool4.get_shape(), out_w=45, out_h=60, scale=2, name='upsample4')
        upsample4 = self.deconv_layer(pool4, [2, 2, 64, 64], [self.config.BATCH_SIZE, 64, 64, 64], 2, "up4")
        # decode 4
        conv_decode4 = self.conv_layer_with_bn(upsample4, [7, 7, 64, 64], self.phase_train, False, name="conv_decode4")

        # upsample 3
        # upsample3 = upsample_with_pool_indices(conv_decode4, pool3_indices, conv_decode4.get_shape(), scale=2, name='upsample3')
        upsample3 = self.deconv_layer(conv_decode4, [2, 2, 64, 64], [self.config.BATCH_SIZE, 128, 128, 64], 2, "up3")
        # decode 3
        conv_decode3 = self.conv_layer_with_bn(upsample3, [7, 7, 64, 64], self.phase_train, False, name="conv_decode3")

        # upsample2
        # upsample2 = upsample_with_pool_indices(conv_decode3, pool2_indices, conv_decode3.get_shape(), scale=2, name='upsample2')
        upsample2 = self.deconv_layer(conv_decode3, [2, 2, 64, 64], [self.config.BATCH_SIZE, 256, 256, 64], 2, "up2")
        # decode 2
        conv_decode2 = self.conv_layer_with_bn(upsample2, [7, 7, 64, 64], self.phase_train, False, name="conv_decode2")

        # upsample1
        # upsample1 = upsample_with_pool_indices(conv_decode2, pool1_indices, conv_decode2.get_shape(), scale=2, name='upsample1')
        upsample1 = self.deconv_layer(conv_decode2, [2, 2, 64, 64], [self.config.BATCH_SIZE, 512, 512, 64], 2, "up1")
        # decode4
        conv_decode1 = self.conv_layer_with_bn(upsample1, [7, 7, 64, 64], self.phase_train, False, name="conv_decode1")

        """ Start Classify """
        # output predicted class number (6)
        with tf.variable_scope('conv_classifier',  reuse=tf.AUTO_REUSE) as scope:
            kernel = util._variable_with_weight_decay('weights',
                                                 shape=[1, 1, 64, 2],
                                                 initializer=customer_init.msra_initializer(1, 64),
                                                 wd=0.0005)
            conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = util._variable('biases', [2], tf.constant_initializer(0.0))
            conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

        logit = conv_classifier

        loss = self.cal_loss(conv_classifier, self.train_label_node)

        return loss, logit

    def cal_loss(self, conv_classifier, labels):
        with tf.name_scope("loss"):
            logits = tf.reshape(conv_classifier, (-1, self.config.NUM_CLASSES))
            epsilon = tf.constant(value=1e-10)
            logits = logits + epsilon
            softmax = tf.nn.softmax(logits)

            # consturct one-hot label array
            label_flat = tf.reshape(labels, (-1, 1))


            # should be [batch ,num_classes]
            labels = tf.reshape(tf.one_hot(label_flat, depth=self.config.NUM_CLASSES), (-1, self.config.NUM_CLASSES))


            w1_n = tf.ones([softmax.shape[0],1],tf.float32)
            w2_n = tf.slice(softmax,[0,0],[-1,1])

            _T = 0.3

            T = tf.ones(softmax.shape[0],1) * _T

            condition = tf.greater(w2_n, 0.5)

            w2_n = tf.where(condition, tf.math.maximum(_T, w2_n), tf.ones(w2_n.shape))

            #w2_n = tf.cond(tf.greater(w2_n, 0.5), lambda : 1-w2_n, lambda : [1])
            #tf.cond(tf.greater(w2_n,0.5) , lambda : 1, lambda : 0)


            weight = tf.concat([w2_n,w1_n],1)

            cross_entropy = -tf.reduce_sum(weight * labels * tf.log(softmax + epsilon), axis=[1])

            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

            tf.add_to_collection('losses', cross_entropy_mean)

            loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        return loss


    def conv_layer_with_bn(self, inputT, shape, train_phase, activation = True, name = None):
        in_channel = shape[2]
        out_channel = shape[3]
        k_size = shape[0]
        with tf.variable_scope(name,  reuse=tf.AUTO_REUSE) as scope:
            kernel = util._variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
            conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
            biases = util._variable('biases', [out_channel], tf.constant_initializer(0.0))
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

    def deconv_layer(self, inputT, f_shape, output_shape, stride=2, name=None):
        # output_shape = [b, w, h, c]
        # sess_temp = tf.InteractiveSession()
        sess_temp = tf.global_variables_initializer()
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            weights = self.get_deconv_filter(f_shape)
            deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                            strides=strides, padding='SAME')
        return deconv

    def get_deconv_filter(self, f_shape):
        """
          reference: https://github.com/MarvinTeichmann/tensorflow-fcn
        """
        width = f_shape[0]
        heigh = f_shape[0]
        f = ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="up_filter", initializer=init,
                               shape=weights.shape)

    def get_train_val(self, image_filenames, label_filenames):
        val_size = int(len(image_filenames) * 0.06)
        val_image_filenames = []
        val_label_filenames = []
        for i in range(val_size):
            pop_index = random.randint(0, len(image_filenames)-1)
            val_image_filenames.append(image_filenames.pop(pop_index))
            val_label_filenames.append(label_filenames.pop(pop_index))
        val_image_filenames.pop(0)
        val_label_filenames.pop(0)
        return image_filenames, label_filenames, val_image_filenames, val_label_filenames

    def training(self, is_finetune=False):
        batch_size = self.config.BATCH_SIZE
        train_dir = self.config.log_dir  # ../data/Logs
        image_dir = self.config.image_dir  # ../data/train
        val_dir = self.config.val_dir  # ../data/val
        finetune_ckpt = self.config.finetune
        image_w = self.config.IMAGE_WIDTH
        image_h = self.config.IMAGE_HEIGHT
        image_c = self.config.IMAGE_DEPTH
        image_filenames, label_filenames = readfile.get_filename_list(image_dir, prefix = "../data/train")
        image_filenames.pop(0)
        label_filenames.pop(0)
        image_filenames.pop(0)
        label_filenames.pop(0)
        print "total file size {}".format(len(image_filenames))
        #val_image_filenames, val_label_filenames = readfile.get_filename_list(val_dir, prefix = "../data/val", is_train=False)

        # image_filenames, label_filenames, val_image_filenames, val_label_filenames = self.get_train_val(image_filenames, label_filenames)
        # print "train size {}".format(len(image_filenames))
        # print "test size {}".format(len(val_image_filenames))


        # should be changed if your model stored by different convention
        startstep = 0 if not is_finetune else int(self.config.finetune.split('-')[-1])

        #with tf.device('/device:GPU:0'):
        with tf.Graph().as_default():
            self.add_placeholders()
            self.global_step = tf.Variable(0, trainable=False)

            train_dataset = readfile.get_dataset(image_filenames, label_filenames, self.config.BATCH_SIZE, True)

            # val_dataset = readfile.get_dataset(val_image_filenames, val_label_filenames, self.config.EVAL_BATCH_SIZE)

            train_iterator = train_dataset.make_one_shot_iterator()
            next_train_element = train_iterator.get_next()

            # val_iterator = val_dataset.make_one_shot_iterator()
            # next_val_element = val_iterator.get_next()

            # Build a Graph that computes the logits predictions from the inference model.
            loss, eval_prediction = self.add_prediction_op()
            # Build a Graph that trains the model with one batch of examples and updates the model parameters.
            train_op = self.add_training_op(loss)
            saver = tf.train.Saver(tf.global_variables(),write_version= saver_pb2.SaverDef.V1)
            summary_op = tf.summary.merge_all()
            with tf.Session() as sess:
                # Build an initialization operation to run below.
                if (is_finetune == True):
                    saver.restore(sess, finetune_ckpt)
                else:
                    init = tf.global_variables_initializer()
                    sess.run(init)

                # Summery placeholders
                summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
                average_pl = tf.placeholder(tf.float32)
                acc_pl = tf.placeholder(tf.float32)
                iu_pl = tf.placeholder(tf.float32)
                average_summary = tf.summary.scalar("test_average_loss", average_pl)
                acc_summary = tf.summary.scalar("test_accuracy", acc_pl)
                iu_summary = tf.summary.scalar("Mean_IU", iu_pl)
                for step in range(startstep, startstep + self.config.maxsteps):
                    image_batch, label_batch = sess.run(next_train_element)
                    # since we still use mini-batches in validation, still set bn-layer phase_train = True
                    feed_dict = {
                        self.train_data_node: image_batch,
                        self.train_label_node: label_batch,
                        self.phase_train: True
                    }
                    start_time = time.time()

                    _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                    duration = time.time() - start_time

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    if step % 10 == 0:
                        num_examples_per_step = batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)

                        format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                                      'sec/batch)')
                        print (format_str % (datetime.datetime.now(), step, loss_value,
                                             examples_per_sec, sec_per_batch))

                        # eval current training batch pre-class accuracy
                        pred = sess.run(eval_prediction, feed_dict=feed_dict)
                        util.per_class_acc(pred, label_batch)

                    # if step % 100 == 0:
                    #     print("start validating.....")
                    #     total_val_loss = 0.0
                    #     hist = np.zeros((self.config.NUM_CLASSES, self.config.NUM_CLASSES))
                    #     for test_step in range(int(self.config.TEST_ITER)):
                    #         val_images_batch, val_labels_batch = sess.run(next_val_element)
                    #
                    #         _val_loss, _val_pred = sess.run([loss, eval_prediction], feed_dict={
                    #             self.train_data_node: val_images_batch,
                    #             self.train_label_node: val_labels_batch,
                    #             self.phase_train: True
                    #         })
                    #         total_val_loss += _val_loss
                    #         hist += util.get_hist(_val_pred, val_labels_batch)
                    #     print("val loss: ", total_val_loss / self.config.TEST_ITER)
                    #     acc_total = np.diag(hist).sum() / hist.sum()
                    #     iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
                    #     test_summary_str = sess.run(average_summary, feed_dict={average_pl: total_val_loss / self.config.TEST_ITER})
                    #     acc_summary_str = sess.run(acc_summary, feed_dict={acc_pl: acc_total})
                    #     iu_summary_str = sess.run(iu_summary, feed_dict={iu_pl: np.nanmean(iu)})
                    #     util.print_hist_summery(hist)
                    #     print(" end validating.... ")
                    #
                    #     summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    #     summary_writer.add_summary(summary_str, step)
                    #     summary_writer.add_summary(test_summary_str, step)
                    #     summary_writer.add_summary(acc_summary_str, step)
                    #     summary_writer.add_summary(iu_summary_str, step)
                    # Save the model checkpoint periodically.
                    if step % 1000 == 0 or (step + 1) == self.config.maxsteps:
                        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)

    def get_submission_result(self, meta_name = None, data_name = None):
        is_first = True

        with tf.Session() as sess:
            self.add_placeholders()

            prediction = np.random.randint(2, size=self.train_label_node.shape)
            prediction.astype(np.float32)

            loss, eval_prediction = self.add_prediction_op()
            # meta_file_path = os.path.join(self.config.test_ckpt, meta_name)
            # if os.path.isfile(meta_file_path):
            #     saver = tf.train.import_meta_graph(meta_file_path,clear_devices=True)
            # else:
            #     raise Exception('restore graph meta data fail')

            saver = tf.train.Saver()
            data_file_path = os.path.join(self.config.test_ckpt, data_name)
            if os.path.isfile(data_file_path):
                saver.restore(sess, data_file_path)
            else:
                raise Exception('restore variable data fail')
            #chkp.print_tensors_in_checkpoint_file(data_file_path, tensor_name = '', all_tensors = True)
            image_filenames, label_filenames = readfile.get_filename_list("../data/val", prefix="../data/val", is_train=False)


            # the length of validation set; 2169
            print "image length {}".format(len(image_filenames))
            # construct the image dataset
            image_paths = tf.convert_to_tensor(image_filenames, dtype=tf.string)
            dataset = tf.data.Dataset.from_tensor_slices(image_paths)
            dataset = dataset.map(readfile.map_fn_test, num_parallel_calls=8)
            dataset = dataset.batch(self.config.BATCH_SIZE)

            test_iterator = dataset.make_one_shot_iterator()
            test_next_element = test_iterator.get_next()

            for i in range(len(image_filenames)/self.config.BATCH_SIZE):
            #for i in range(2):
                # for i in range(len(image_filenames))
                image_batch = sess.run(test_next_element)

                #print image_batch.shape

                feed_dict = {
                    self.train_data_node: image_batch,
                    self.phase_train: True
                }

                if is_first:
                    result = sess.run([eval_prediction],feed_dict)[0]
                    # prediction = tf.stack([prediction, result])
                    print "prediction shape : {}".format(result.shape)
                    is_first = False
                    continue
                # 5,512,512,2
                new_result = sess.run([eval_prediction],feed_dict)[0]
                #print "old result shape {}".format(np.asarray(result).shape)
                #print "new result shape {}".format(new_result.shape)
                result = np.concatenate([result, new_result],axis=0)

                #prediction = tf.stack([prediction, result])
                print "prediction shape : {}".format(result.shape)

            # for i in range(self.config.BATCH_SIZE):
            #     util.writemask(result[1][i],'mask_'+str(i)+".png")
            # preprocess the prediction and product submission, prediction is [numexample, 512, 512, 2]
            util.create_submission('../data/subid2_1.csv', result, image_filenames)

if __name__ == '__main__':
    segmodel = SegnetModel()
    # print all tensors in checkpoint file
    segmodel.get_submission_result(meta_name="model.ckpt-19000.meta", data_name="model.ckpt-19000")

