# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys


import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

#Original function to download the data from YLC's website
#from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from mnist_mine import read_data_sets_microspheres

FLAGS = None


def conv2d(input, channels_in, channels_out, name = "conv2d"):
	with tf.name_scope(name):
		w = tf.Variable(tf.truncated_normal([6,6, channels_in, channels_out], stddev = 0.1), name = "W")
		b = tf.Variable(tf.constant(0.1, shape=[channels_out], name = "B"))
		conv = tf.nn.conv2d(input, w, strides= [1,1,1,1], padding="SAME")
		act = tf.nn.relu(conv + b)
		tf.summary.histogram("weights", w)
		tf.summary.histogram("biases", b)
		tf.summary.histogram("activations", act)
		return act
def conv2d_pool(input, channels_in, channels_out, name = "conv_pool"):
	with tf.name_scope(name):
		w = tf.Variable(tf.truncated_normal([5,5, channels_in, channels_out], stddev = 0.1), name = "W")
		b = tf.Variable(tf.constant(0.1, shape = [channels_out]), name="B")
		conv = tf.nn.conv2d(input, w, strides=[1,2,2,1], padding="SAME")
		act = tf.nn.relu( conv + b)
		tf.summary.histogram('weights', w)
		tf.summary.histogram('biases', b)
		tf.summary.histogram('activations', act)
		return act

def fc_layer(input, channels_in, channels_out, name="fc_layer"):
	with tf.name_scope(name):
		w = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1), name="W")
		b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name="B")
		act = (tf.matmul(input, w) + b)
		tf.summary.histogram('weights', w)
		tf.summary.histogram('biases', b)
		tf.summary.histogram('activations', act)
		return act


##################################################################################################
## MODEL now

def mnist_model(learning_rate):
	# reset the computational graph for every learning rate available
	tf.reset_default_graph()
	sess = tf.Session()
	# create a model now

	image_size = 784
	image_size2D = 28
	num_FP_images = 25

	# input
	x = tf.placeholder(tf.float32, [None, num_FP_images * image_size])
	print('shape of data loaded from MNIST_MINE')
	print(x.get_shape())

	y = tf.placeholder(tf.float32, [None,5]) # Number of classes i have is 5

	x_added_vec = tf.reshape(x, [-1, image_size, num_FP_images])
	print('check size of reshaped images: should be [? , 784, 25]')
	print(x_added_vec.get_shape())

	# define the illumination weight as a variable
	illumination = tf.Variable(tf.truncated_normal([num_FP_images, 1]))
	print('shape of the illumination field')
	print(illumination.get_shape())

	#use einsum to do the mulitplication of 2D slices of th 3D tensor with the 1D vector
	x_added = tf.einsum('aij,jk->aik', x_added_vec, illumination)
	print('shape after einsum multiplication')
	print(x_added.get_shape())

	# reshape the image now to 28 x 28
	x_image_pre = tf.reshape(x_added, [-1, image_size2D,image_size2D])
	x_image = tf.reshape(x_image_pre, [-1, 28, 28, 1])

	print(' final shape of the image and data type')
	print(x_image.get_shape())
	print(x_image.dtype)


	##############################################################################################################################


	##################################################################################################

	# First conv layer
	conv1 = conv2d(x_image, 1, 6, "conv1") # Output should be 28x28x6

	# 2nd layer
	conv2 = conv2d_pool(conv1, 6, 12, "conv_pool") # output should be 14x14x12

	# 3rd conv layer 
	conv3 = conv2d_pool(conv2, 12, 24, "conv_pool_2")# output should be 7x7x24

	# Fully connected layer
	# first reshape the above layer
	flattened = tf.reshape(conv3, [-1, 7*7*24])

	# 1st FC layer
	fc_1 = tf.nn.relu(fc_layer(flattened, 7*7*24, 200, "fc_1"))

	# add dropout
	keep_prob = tf.placeholder(tf.float32)
	fc_drop = tf.nn.dropout(fc_1, keep_prob)

	# Final layer : also 2nd fc layer
	y_ = fc_layer(fc_drop, 200, 5, "classification_layer")

	

	################################################################################

	################################################################################

	# LOSS and TRAINING
	# use cross entropy as the loss function
	with tf.name_scope("X-entropy"):
		cross_entropy = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_))
		tf.summary.scalar('cross_entropy', cross_entropy)
		# Use ADAM optimizer for training
		with tf.name_scope("train"):
			train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

	# compute accuracy
	with tf.name_scope("accuracy"):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy', accuracy)

	# import the data
	mnist = read_data_sets_microspheres("data/", one_hot=True)

	#define a feeding in dictionary to load in the data: very efficient
	def feed_dict(train):
		if train or FLAGS.fake_data:
			xs, ys = mnist.train.next_batch(50, fake_data=FLAGS.fake_data)
			k = FLAGS.dropout
		else:
			xs, ys = mnist.test.images, mnist.test.labels
			k = 1.0
		return{x: xs, y: ys, keep_prob:k}

	#train the cnn now inside a loop for specified number of iterations
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# merger all th summaries
		merged_summary = tf.summary.merge_all()

		# use a log directory to store the summaries to disk for use in TB
		train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
		test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')


		for i in range(15000):
			if i % 100==0:
				s = sess.run(merged_summary, feed_dict=feed_dict(True))
				train_writer.add_summary(s, i)
				train_accuracy = accuracy.eval(feed_dict=feed_dict(True))

				print('step %d, training accuracy %g' % (i, train_accuracy))
			train_step.run(feed_dict=feed_dict(True))

		print('test accuracy %g' % accuracy.eval(feed_dict=feed_dict(False)))
		s = sess.run(merged_summary, feed_dict=feed_dict(True))
		test_writer.add_summary(s, i)


def main(_):
	for learning_rate in [1e-3]:
		print('START RUN FOR LEARNING RATE %s ' % learning_rate)
		mnist_model(learning_rate)
		print('FINISHED RUN FOR LEARNING RATE %s' % learning_rate)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, use fake data for unit testing')
  parser.add_argument('--dropout', type=float, default=0.5,
                      help='Keep probability for training dropout')
  parser.add_argument('--log_dir', type=str,
                      default=r'C:\Users\Roarke\Desktop\microspheres simulations\classification\tensorboard',
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)







































