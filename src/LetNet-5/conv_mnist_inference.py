
# -*- coding: utf-8 -*-
# -*- version: 
#		python 3.5.2
#		tensorflow 1.4.0
#		numpy 1.13.1
# 
# -*- ---------------------------------- -*-
#
# conv_mnist_inferenct.py 
#	-- Define forward propagation progress and arguments of LeNet-5 .
import tensorflow as tf
import numpy as np

# Neural Network Arguments .
INPUT_NODE = 784
OUTPUT_NODE = 10

# Input samples arguments .
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# Convolutional layer1 arguments .
CONV1_DEEP = 32
CONV1_SIZE = 5
# Convolutional layer2 arguments .
CONV2_DEEP = 64
CONV2_SIZE = 5
# Full-connected layer arguments .
FC_SIZE = 512


# Forward Propagation of LeNet-5 .
#
# Argument train added to distinguish training and testing progress ;
# Dropout method added to improve model reliability and prevent from over-fitting ;
# which just used in training process .
def inference(input_tensor, train, regularizer):
	# Declare convolutional layer1 variables and forward-propagation in different
	# namescope , in which variable name defined without care about rename trouble .
	# 
	# Convolutional layer1 output 28*28*32 with 28*28*1 image pixel 
	# data input using zero-padding .
	with tf.variable_scope('layer1-conv1'):
		# Convolutional filter size = 5*5*64 .
		conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE,
							   NUM_CHANNELS, CONV1_DEEP],
						initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

		# Filter with size=5*5 , deepth=32 , stride=1 with zero-padding .
		conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1,1,1,1], padding='SAME')
		relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))


	# Declare pool layer2 variables and forward-propagation ,
	#
	# Max-pool filter used with size=2*2 , zero-padding , and stride=2 ,
	# convolutional layer1 output as pool layer2 input and output 14*14*32 matrix .
	with tf.name_scope('layer2-pool1'):
		pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1],
				strides=[1, 2, 2, 1], padding='SAME')

	# Declare convolutional layer3 and forward-propagation , with 14*14*32 matrix input
	# and 14*14*64 matrix output .
	with tf.variable_scope('layer3-conv2'):
		conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE,
							   CONV1_DEEP, CONV2_DEEP],
						initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

		# Filter with size=5*5, deepth=64 , stride=1 with zero-padding .
		conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

	
	# Declare pool layer4 and forward-propagation , with 14*14*64 input and 7*7*64 output .
	with tf.name_scope('layer4-pool2'):
		pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1],
				strides=[1, 2, 2, 1], padding='SAME')

	
		# Transform pool2 output 7*7*64 matrix into vector as input of full-connected layer5 
		# using API get_shape() to get dimension rather than mannual computing . 
		#
		# Every layer of network inputs a batch of data , hence the dimension information
		# here is for a batch . 
		pool_shape = pool2.get_shape().as_list()
	
		# Calculating the length of transforming vector , and pool_shape[0] is the number of data in one batch .
		nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]

		# Reshape output of layer4 into a vector including one batch of data .
		reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

	# Declare full-connected layer5 and forward-propagation , 
	# with a group of vectors (length=3136) input , and vector (length=512) output .
	# Dropout used to change part of output to zeros randomly ,
	# which usually used on full-connected layer rather than convolutional and pool layer .

	with tf.variable_scope('layer5-fc1'):
		fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
				initializer=tf.truncated_normal_initializer(stddev=0.1))
		# Regularizer in full-connected layer .
		if regularizer != None:
			tf.add_to_collection('losses', regularizer(fc1_weights))
		fc1_biases = tf.get_variable("bias", [FC_SIZE],
				initializer=tf.constant_initializer(0.1))
		fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
		
		# Add dropout in full-connected layer when training .
		if train:
			fc1 = tf.nn.dropout(fc1, 0.5)

	# Declare full-connected layer6 and forward-propagation ,
	# with vector (length=512) input and output vector (length=10) ,
	# through which we can get classification result by adding softmax at last .
	with tf.variable_scope('layer6-fc2'):
		fc2_weights = tf.get_variable('weight', [FC_SIZE, NUM_LABELS],
				initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None:
			tf.add_to_collection("losses", regularizer(fc2_weights))
		fc2_biases = tf.get_variable("bias", [NUM_LABELS],
				initializer=tf.constant_initializer(0.1))
		
		logit = tf.matmul(fc1, fc2_weights) + fc2_biases

	# Get output of full-connected layer6
	return logit





































