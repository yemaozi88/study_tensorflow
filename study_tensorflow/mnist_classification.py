from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# user define
dir_curr = r'c:\Users\A.Kunikoshi\source\repos\study\study'
data_dir = dir_curr + "\\MNIST_data"


# import data
mnist = input_data.read_data_sets(data_dir, one_hot=True)
#mnist = input_data.read_data_sets(data_dir)


# place holder for the input: variable length x 784 (28 x 28)
x = tf.placeholder(tf.float32, [None, 784], 'x')

# initialize weight and bias with 0.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# model
#y = tf.nn.softmax(tf.matmul(x, W) + b)
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.int64, [None], 'y_')


# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#
# So here we use tf.losses.sparse_softmax_cross_entropy on the raw
# outputs of 'y', and then average across the batch.
cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
# learning rate = 0.5
train_step	  = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# train model
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(1000):
	# choose 100 samples randomly
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
