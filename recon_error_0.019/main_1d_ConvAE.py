#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 09:49:19 2017

@author: cyril
"""
from conv2d import Conv2d
from max_pool_2d import MaxPool2d

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
import tensorflow as tf
import numpy as np

# Tensorboard /home/dawei/anaconda2/envs/lstm/lib/python3.5/site-packages/tensorflow/tensorboard/tensorboard.py --logdir=path/to/log-directory

@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolWithArgmaxGrad(op, grad, unused_argmax_grad):
    return gen_nn_ops._max_pool_grad(op.inputs[0],
                                     op.outputs[0],
                                     grad,
                                     op.get_attr("ksize"),
                                     op.get_attr("strides"),
                                     padding=op.get_attr("padding"),
                                     data_format='NHWC')


class Network:


    def __init__(self, layers = None, batch_norm=True, skip_connections=True, pretrain= False):
        # Define network - ENCODER (decoder will be symmetric).
        self.IMAGE_HEIGHT = 2048
        self.IMAGE_WIDTH = 1
        self.IMAGE_CHANNELS = 1
        
        if pretrain:
            # load pretrained weights from TICNN
            pre_train_path = '/home/dawei/cyril/1d_conv_encoder_decoder/pretrained_TICNN/'
            W_conv1 = tf.constant(np.load(pre_train_path + 'W_conv1.npy'))
            W_conv2 = tf.constant(np.load(pre_train_path + 'W_conv2.npy'))
            W_conv3 = tf.constant(np.load(pre_train_path + 'W_conv3.npy'))
            W_conv4 = tf.constant(np.load(pre_train_path + 'W_conv4.npy'))
            W_conv5 = tf.constant(np.load(pre_train_path + 'W_conv5.npy'))
        else:
            W_conv1 = None
            W_conv2 = None
            W_conv3 = None
            W_conv4 = None
            W_conv5 = None
        
        if layers == None:
            layers = []
            layers.append(Conv2d(kernel_size=64, strides=[1, 8, 1, 1], output_channels=16, name='conv_1', initializer = W_conv1))
            layers.append(MaxPool2d(kernel_size=[2, 1], name='max_1',skip_connection=True and skip_connections))
#
            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=32, name='conv_2', initializer = W_conv2))  
            layers.append(MaxPool2d(kernel_size=[2, 1], name='max_2',skip_connection=True and skip_connections))
#
            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=64, name='conv_3', initializer = W_conv3))
            layers.append(MaxPool2d(kernel_size=[2, 1], name='max_3',skip_connection=True and skip_connections))
            
            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=64, name='conv_4', initializer = W_conv4))
            layers.append(MaxPool2d(kernel_size=[2, 1], name='max_4',skip_connection=True and skip_connections))

            layers.append(Conv2d(kernel_size=3, strides=[1, 1, 1, 1], output_channels=64, name='conv_5', initializer = W_conv5))
            layers.append(MaxPool2d(kernel_size=[2, 1], name='max_5'))
            

        self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS],
                                     name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')
        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
        self.description = ""

        self.layers = {}

        net = self.inputs   #(None, 2048, 1, 1)
        
        # ENCODER
        for layer in layers:
            self.layers[layer.name] = net = layer.create_layer(net)
            self.description += "{}".format(layer.get_description())

        print("Current input shape: ", self.inputs.get_shape())
        
        layers.reverse()   # reverse the list of layers
        Conv2d.reverse_global_variables()

        # DECODER
        for layer in layers:
            net = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name])

        #self.segmentation_result = tf.sigmoid(net)
        self.segmentation_result = net
        print('segmentation_result.shape: {}, targets.shape: {}'.format(self.segmentation_result.get_shape(),
                                                                        self.targets.get_shape()))

        # RMSE loss
        self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.segmentation_result - self.targets)))
        self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.cost)

        tf.summary.scalar('rmse_cost', self.cost)
        self.summary = tf.summary.merge_all()

class Dataset:
    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size

        self.data = np.load(data_path + 'data_channel1_minmax_norm.npy')  # 175200, 2048
        self.data = self.data.reshape((self.data.shape[0], self.data.shape[1], 1, 1))   # 175200, 2048, 1, 1
        
        
    def gen_batch(self, shuffle=True):

        data_length = self.data.shape[0]
        if shuffle:
            indices = np.arange(data_length)
            np.random.shuffle(indices)
        for start_idx in range(0, data_length - self.batch_size + 1, self.batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + self.batch_size]
            else:
                excerpt = slice(start_idx, start_idx + self.batch_size)
            batch_raw_data = self.data[excerpt]   # (batch_size, num_steps, num_features+1)
            yield batch_raw_data
        
    def gen_epochs(self, max_epoch):
        for i in range(max_epoch):
            yield self.gen_batch(shuffle = True) # the input data is padded

def train():
    BATCH_SIZE = 300
    max_epoch = 600
    
    
    network = Network(pretrain=False)
    dataset = Dataset(data_path = './IMS_BearingData/',
                      batch_size=BATCH_SIZE)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('./log/conv_ae' + '/train',sess.graph)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        
        for i, epoch in enumerate(dataset.gen_epochs(max_epoch)):
            print("Epoch: %d/%d..." % (i + 1, max_epoch))
            step = 0  # the number of mini batch training in a epoch
            for batch_x in epoch:  # randomly select mini batch from training data
                step += 1
                loss, _ ,summary= sess.run([network.cost, network.train_op, network.summary],
                                              feed_dict={network.inputs: batch_x, network.targets: batch_x,
                                              network.is_training: True})
                train_writer.add_summary(summary, step)
                if step % 10 == 0:
                    print("epoch {}, step {}, training loss {}".format(i+1, step, loss))
            saver.save(sess, './log/check_point/model_epoch', global_step=i+1)
if __name__ == '__main__':
    train()