
# -*- coding: utf-8 -*-
# -*- version:
#		python 3.5.3
#		tensorflow 1.4.0 
#		numpy 1.13.1
# -*- ---------------------------------- -*-  
#
# Raw version from Tensorflow battle in Google Framework 
# modified by Hsingmin Lee to update new features and debug
# the Model to runnable .

# rnn_language_model.py -- Achieve a 
# natrual language processing model with deepRNN and LSTM .

import numpy as np
import tensorflow as tf
import tensorflow_ptb_reader as reader

# Constants of data
DATA_PATH = "./ptb/data"		# Dataset stored path
HIDDEN_SIZE = 200			# Hidden layer nodes 
NUM_LAYERS = 2				# deepRNN LSTM layers
VOCAB_SIZE = 10000			# Dictionary size .

# Constants of neural network
LEARNING_RATE = 1.0			# Learning rate in training process
TRAIN_BATCH_SIZE = 20			# Input data batch size
TRAIN_NUM_STEP = 35			# Training data truncate length

# Regard test data as a super long sequence for no truncating used in test process .
EVAL_BATCH_SIZE = 1			# Test data batch size
EVAL_NUM_STEP = 1			# Test data truncate length
NUM_EPOCH = 2				# Epoches of using test data
KEEP_PROB = 0.5				# Probability of no dropout for one node
MAX_GRAD_NORM = 5			# Coefficient to control gradient expansion 

# Create PTBModel to describe model and maintain state in RNN ,
# and defines ops for build neural network .
class PTBModel(object):
	def __init__(self, is_training, batch_size, num_steps):
		# Record batch size and truncate length
		self.batch_size = batch_size
		self.num_steps = num_steps

		# Define input layer with size=batch_size*num_steps ,
		# which equals to the batch size output by ptb_iterator .
		self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])

		# Define expected output with size equals to real label output by ptb_iterator.
		self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

		# Set LSTM to be loop structure of deepRNN and using dropout .
		# Set state_is_tuple=True , returns (c, h) or it would be concated into a tensor .
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
		if is_training:
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
					lstm_cell, output_keep_prob=KEEP_PROB)
		
		cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS, state_is_tuple=True)

		# Initialize original state to zeros vector .
		self.initial_state = cell.zero_state(batch_size, tf.float32)
		# Converse word id to word vector .
		#
		# Words counts totally to VOCAB_SIZE in dictionary ,
		# word vector dimension=HIDDEN_SIZE , then variable embedding
		# dimension=VOCAB_SIZE * HIDDEN_SIZE .
		
		embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

		# Converse original batch_size * num_steps words' id into word vector.
		# word vector dimension = batch_size x num_steps x HIDDEN_SIZE ,
		# in which batch_size as the first dimenion ,
		# num_steps as the second dimension,
		# HIDDEN_SIZE as the third dimension .
		# Get a 3-D word matrix .
		inputs = tf.nn.embedding_lookup(embedding, self.input_data)

		# Use dropout only in training process .
		if is_training:
			inputs = tf.nn.dropout(inputs, KEEP_PROB)
		
		# Define outputs array to collect LSTM output in different moment ,
		# and get the final output through a full-connected network .
		outputs = []
		# Store LSTM state information of different batch , and initialize to zeros .
		state = self.initial_state
		
		with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
			for time_step in range(num_steps):
				if time_step > 0:
					tf.get_variable_scope().reuse_variables()
				# Input training data reshaped in embedding following sequence .
				cell_output, state = cell(inputs[:, time_step, :], state)

				outputs.append(cell_output)
		# Reshape output into input matrix dimension .
		#
		# In tensorflow 1.0 or higher version , change tf.concat(1, outputs)
		# to tf.concat(outputs, 1) .
		output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
		
		# Final full-connected layer to get the predication value ,
		# that is an array length=VOCAB_SIZE , which turned to be a
		# probability vector through softmax layer .

		weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
		bias = tf.get_variable("bias", [VOCAB_SIZE])
		logits = tf.matmul(output, weight) + bias

		# Cross Entropy loss function ,
		# Tensorflow provides sequence_loss_by_example api to calculate 
		# the cross-entropy of one sequence .
		#
		# In tensorflow 1.0 or higher version , tf.nn.seq2seq.sequence_loss_by_example
		# is romoved and use tf.contrib.legacy_seq2seq.sequence_loss_by_example() instead .
		loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
				[logits],				# Predication value
				[tf.reshape(self.targets, [-1])],	# Expected result 
									# reshape [batch_size, num_steps] 
									# array into one list 
				# Loss weight set to be 1 , which means the loss of 
				# different batch on different moment matters the same importance .  
				[tf.ones([batch_size * num_steps], dtype=tf.float32)])
		# Calculate loss of every batch .
		self.cost = tf.reduce_sum(loss) / batch_size
		self.final_state = state
		
		if not is_training:
			return
		trainable_variables = tf.trainable_variables()
		# Control gradient with tf.clip_by_global_norm() to avoid gradient explosion .
		grads, _ = tf.clip_by_global_norm(
				tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)

		# Define optimizer .
		optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)

		# Define training steps .
		self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

# Batch all text contents for model to train .
# Return perplexity value on whole dataset by running train_op .
def run_epoch(session, model, data, train_op, output_log):
	# Auxiliary variables to calculate perplexity .
	total_costs = 0.0
	iters = 0
	state = session.run(model.initial_state)
	count = 0
	# Train or test model with current dataset .
	for step, (x, y) in enumerate(
			reader.ptb_iterator(data, model.batch_size, model.num_steps)):
		# Run train_op on current batch and calculate cross-entropy that means
		# the probability of next word been specified .
		cost, state, _ = session.run([model.cost, model.final_state, train_op],
				{model.input_data: x, model.targets: y, model.initial_state: state})

		# Add probability of all batched in all moments to get perplexity .
		total_costs += cost
		iters += model.num_steps
		count += 1
		# Output log when training .
		if output_log and count % 100 == 1:
			print("After %d steps, perplexity is %.3f " %(
				count, np.exp(total_costs / iters)))

	# Return perplexity on training dataset .
	return np.exp(total_costs / iters)

# Calls run_epoch() for many times ,
# contents in text will be feeded to model for many times ,
# in which progress the arguments adjusted . 
def main(argv=None):
	# Get raw training dataset .
	train_data, validate_data, test_data, _ = reader.ptb_raw_data(DATA_PATH) 
	
	# Define initializer for model .
	initializer = tf.random_uniform_initializer(-0.05, 0.05)
	
	# Define deepRNN model for training .
	with tf.variable_scope("language_model", 
			reuse=tf.AUTO_REUSE, initializer=initializer):
		train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

	# Define deepRNN model for testing .
	with tf.variable_scope("language_model", 
			reuse=tf.AUTO_REUSE, initializer=initializer):
		eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)
	
	with tf.Session() as session:
		session.run((tf.global_variables_initializer(),
			  tf.local_variables_initializer()))

		# Train model with training datasets
		for i in range(NUM_EPOCH):
			print("In iteration: %d " % (i + 1))
			# Train deepRNN model on whole dataset .
			run_epoch(session, train_model, 
					train_data, train_model.train_op, True)
			
			# Validate model with validate dataset .
			validate_perplexity = run_epoch(session, eval_model,
					validate_data, tf.no_op(), False)
			print("Epoch: %d Validation Perplexity %.3f" % 
					(i + 1, validate_perplexity))

		
		# Evaluate model performance on test dataset .
		test_perplexity = run_epoch(session, eval_model, test_data, tf.no_op(), False)
		print("Test: Perplexity: %.3f " % test_perplexity)

if __name__ == "__main__":
	tf.app.run()














































