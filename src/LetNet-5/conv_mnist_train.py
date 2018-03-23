
# conv_mnist_train.py -- Training Neural Network with LeNet-5 .
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# Load constants and forward propagation function in conv_mnist_inference.py
import conv_mnist_inference

# Network Arguments
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# Model saved path .
MODEL_SAVE_PATH = './model/'
MODEL_NAME = "model.ckpt"

def train(mnist):
	# Define inputs placeholder which is a 4-dimension matrix .
	#
	# [samples_in_batch, sample_length, sample_width, sample_depth]
	x = tf.placeholder(tf.float32, [BATCH_SIZE,
					conv_mnist_inference.IMAGE_SIZE,
					conv_mnist_inference.IMAGE_SIZE,
					conv_mnist_inference.NUM_CHANNELS],
					name='x-input')
	y_ = tf.placeholder(tf.float32, [BATCH_SIZE, conv_mnist_inference.OUTPUT_NODE], name='y-input')


	# Regularizer defination .
	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	
	# Forward propagation by calling conv_mnist_inference.inference()
	y = conv_mnist_inference.inference(x, True, regularizer)
	# Record global training steps .
	global_step = tf.Variable(0, trainable=False)

	# Define loss-function , learning-rate, moving-average and training-steps .
	#
	# Apply MovingAverage to all trainable variables to get a robust model . 
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())

	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	# Add cross-entropy and regularizer saved in collection .
	loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

	# learning_rate = LEARNING_RATE_BASE * LEARNING_RATE_DECAY ^ 
	# (global_step / (mnist.train.num_examples/BATCH_SIZE))
	learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
							global_step,
							mnist.train.num_examples/BATCH_SIZE,
							LEARNING_RATE_DECAY, staircase=True)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	
	# Update both variables and the moving-averages .
	# tensorflow.no_op() is a operation placeholder .
	with tf.control_dependencies([train_step, variables_averages_op]):
		train_op = tf.no_op(name='train')

	# Initialize persistence class .
	saver = tf.train.Saver()
	
	with tf.Session() as sess:
		tf.global_variables_initializer().run()

		# Raw training process , no validating and testing .
		for i in range(TRAINING_STEPS):
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			
			# Reshape xs to be a 4-dimension matrix .
			reshaped_xs = np.reshape(xs, (BATCH_SIZE,
						      conv_mnist_inference.IMAGE_SIZE,
						      conv_mnist_inference.IMAGE_SIZE,
						      conv_mnist_inference.NUM_CHANNELS))
			
			_, loss_value, step = sess.run([train_op, loss, global_step],
							feed_dict={x: reshaped_xs, y_: ys})

			# Save model every 1000 steps .
			if i % 1000 == 0:
			# Calculate current loss on training dataset .
				print("After %d training steps, loss on training batch is %g ." %(step, loss_value))

				# Save current model .
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
def main(argv=None):
	mnist = input_data.read_data_sets("./data/", one_hot=True)
	train(mnist)

if __name__ == '__main__':
	tf.app.run()
		





































