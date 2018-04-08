# -*- coding: utf-8 -*-
#
# -*- version:
#		python 3.5.2
#		tensorflow 1.4.0 
#		numpy 1.13.1
# -*- author: Hsingmin Lee
#
# inception_migration.py -- Migration learning with Inception-v3 model .
#
# Regard the former layers except the last full-connected layer as 
# a whole the layer called bottleneck .
# Bottleneck is used to extract image features for the new single
# full-connected layer that needed to be trained to solve another
# classification problem .

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# Bottleneck tensor number in Inception-v3 model .
BOTTLENECK_TENSOR_SIZE = 2048

# Tensor name in Inception-v3 model to give bottleneck layer result .
#
# In Inception-v3 model provided by google , the bottleneck layer name 
# 'pool_3/reshape:0' , that can be accessed by tensor.name during model training . 
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# Tensor name for JPEG image input .
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
# Model saved path .
MODEL_DIR = './model'
# Model name .
MODEL_FILE = 'classify_image_graph_def.pb'

# Save the feature vectors to specified file for training dataset
# would be used for many times . 
CACHE_DIR = './tmp/bottleneck'

# Image data directory .
INPUT_DATA = './data/flower_photos'

# Validation data percentage .
VALIDATION_PERCENTAGE = 10
# Test data percentage .
TEST_PERCENTAGE = 10

# Network arguments setting .
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

# Dispatch training dataset , validation dataset and testing dataset
# from image file .
def create_image_lists(testing_percentage, validation_percentage):
	# Save all images dispatched from data file into dictionary result ,
	# in which key means class name , value is a dictionary including images name.
	result = {}
	# Get all sub-directories in input data path . 
	sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]

	# The first element is root directory that needed to be ignored . 
	#
	# sub_dirs=['./data/flower_photos',
	#           './data/flower_photos/daisy',
	# 	    './data/flower_photos/dandelion',
	#           './data/flower_photos/roses',
	#           './data/flower_photos/sunflowers',
	#           './data/flower_photos/tulips']
	
	# The first element in list is root directory .
	is_root_dir = True
	for sub_dir in sub_dirs:
		if is_root_dir:
			is_root_dir = False
			continue
		
		# Get all valid images in current directory .
		extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
		file_list = []
		dir_name = os.path.basename(sub_dir)
		for extension in extensions:
			# Image full name .
			file_glob = os.path.join(INPUT_DATA, dir_name, '*.'+extension)
			file_list.extend(glob.glob(file_glob))
		if not file_list: continue

		# Get class name through directory information .
		label_name = dir_name.lower()
		# Initialize testing dataset , vlidation dataset and training dataset 
		# in current category .
		training_images = []
		testing_images = []
		validation_images = []
		
		for file_name in file_list:
			# Get clean file name without path as prefix .
			base_name = os.path.basename(file_name)
			chance = np.random.randint(100)
			if chance < validation_percentage:
				validation_images.append(base_name)
			elif chance < (testing_percentage + validation_percentage):
				testing_images.append(base_name)
			else:
				training_images.append(base_name)
		# Place current category data into dictionary result 
		# label_name as the key of list image_lists . 
		result[label_name] = {'dir': dir_name,\
				      'training': training_images,\
				      'testing': testing_images,\
				      'validation': validation_images}
	return result

# Get image directory through label_name , category , index arguments .
# label_name : image class label as 'daisy', 'sunflowers', 'dandelion', 'roses', 'tulips' .
# category : image divided category as 'training', 'testing', 'validation' .
# index : image index number of integer .
# 
# image_lists : full image information .
# image_dir : root directory to save image .
def get_image_path(image_lists, image_dir, label_name, index, category):
	# Get all image information in given class label .
	label_lists = image_lists[label_name]
	
	# Get all image information in given divided category .
	category_list = label_lists[category]

	mod_index = index % len(category_list)
	base_name = category_list[mod_index]
	sub_dir = label_lists['dir']

	# Get image full path including root directory , label directory and image name .
	full_path = os.path.join(image_dir, sub_dir, base_name)
	return full_path

# Get path of feature vector output by Inception-v3 model through label ,
# category and index . 
def get_bottleneck_path(image_lists, label_name, index, category):
	return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'

# Get image feature vector through loading Inception-v3 model pre-trained already . 
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
	# Get image feature tensor by processing input image in bottleneck tensor .  
	bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
	# Converse the 4-dimension matrix output by bottleneck into a feature vector .
	bottleneck_values = np.squeeze(bottleneck_values)
	return bottleneck_values

# Get the feature vector output by Inception-v3 model .
# 
# Search for the feature vector firstly , if not exists then create it .
def get_or_create_bottleneck(sess, image_lists, label_name, index, \
		category, jpeg_data_tensor, bottleneck_tensor): 
	# Get feature vector path corresponding to the image .
	label_lists = image_lists[label_name]
	sub_dir = label_lists['dir']
	# Get image feature vector directory .
	sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
	# Create new directory for image feature vector .
	if not os.path.exists(sub_dir_path):
		os.makedirs(sub_dir_path)
	# Actually is a txt file in CACHE_DIR path .
	bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)
	
	# If the feature vector file not exists , 
	# then compute through Inception-v3 model and save to file .
	if not os.path.exists(bottleneck_path):
		# Get image path .
		image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
		# Get image content from given path image_path .
		# 
		# gfile.FastGFile() operate file like open() .
		image_data = gfile.FastGFile(image_path, 'rb').read()
		# Compute feature vector through Inception-v3 model as bottleneck values.
		bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)

		# Store new feature vector into bottleneck_file .
		bottleneck_string = ','.join(str(x) for x in bottleneck_values)
		with open(bottleneck_path, 'w') as bottleneck_file:
			bottleneck_file.write(bottleneck_string)
	else:
		# Get existed feature vector from bottleneck_file .
		with open(bottleneck_path, 'r') as bottleneck_file:
			bottleneck_string = bottleneck_file.read()
		bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

	return bottleneck_values

# Pick a batch of images to be training dataset .
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many,\
		category, jpeg_data_tensor, bottleneck_tensor):
	bottlenecks = []
	ground_truths = []
	for _ in range(how_many):
		# Get and integer smaller than n_classes as label_index . 
		label_index = random.randrange(n_classes)
		# Get label name indexed by label_index .
		label_name = list(image_lists.keys())[label_index]
		# Produce image_index smaller than 65536 randomly .
		image_index = random.randrange(65536)
		
		# Get image feature vector through Inception-v3 model .
		bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,\
				image_index, category, jpeg_data_tensor, bottleneck_tensor)

		# Keep image label information in ground_truth ,
		# return both bottlenecks list and ground_truths list . 
		ground_truth = np.zeros(n_classes, dtype=np.float32)
		ground_truth[label_index] = 1.0
		bottlenecks.append(bottleneck)
		ground_truths.append(ground_truth)

	return bottlenecks, ground_truths

# Get all test dataset .
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
	bottlenecks = []
	ground_truths = []
	label_name_list = list(image_lists.keys())
	
	# Enumerate all class labels and testing images .
	#
	# label_index, label_name in enumerate(label_name_list):
	# 	0 diasy
	#	1 dandelion
	#	2 roses
	#	3 sunflowers
	#	4 tulips
	for label_index, label_name in enumerate(label_name_list):
		category = 'testing'
		# Get image index and image name (unused) .
		for index, unused_base_name in enumerate(image_lists[label_name][category]):
			bottleneck = get_or_create_bottleneck(\
					sess, image_lists, label_name, index, category,\
					jpeg_data_tensor, bottleneck_tensor)
			ground_truth = np.zeros(n_classes, dtype=np.float32)
			ground_truth[label_index] = 1.0
			bottlenecks.append(bottleneck)
			ground_truths.append(ground_truth)
	
	return bottlenecks, ground_truths


# Migration learning main module .
def main(argv=None):
	# Get all images into image_lists .
	image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
	# Classfication labels of images . 
	n_classes = len(image_lists.keys())

	# Get Inception-v3 model pre-trained by Google 
	# that saved in GraphDef Protocol Buffer which including all tensors
	# variable and op .
	with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	# Load imported model and returns tensor corresponding to input data and 
	# tensor output by bottleneck layer .
	# tf_import_graph_def(graph_def, 
	# 		      input_map=None,
	#		      return_elements=None,
	#                     name=None,
	#                     op_dict=None)

	bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def,\
			return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
	
	# Define a new input for network which also as the feature matrix 
	# produced by Inception-v3 model when image data forward propagate
	# to bottleneck layer .
	bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE],\
			name='BottleneckInputPlaceholder')

	# Define a new input for ground truth of image label .
	ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

	# Create a full-connected layer weight_dimension=[BOTTLENECK_TENSOR_SIZE, n_classes] 
	# to solve image classification problem .
	# 
	# For feature vector extracted by Inception-v3 model more easily to classify ,
	# complex network structure like LeNet5 no more needed .
	with tf.name_scope('final_training_ops'):
		weights = tf.Variable(tf.truncated_normal(\
				[BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
		biases = tf.Variable(tf.zeros([n_classes]))
		logits = tf.matmul(bottleneck_input, weights) + biases
		final_tensor = tf.nn.softmax(logits)

	# Cross-entropy Loss function .
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,\
							labels=ground_truth_input)
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	# Last full-connected layer training .
	train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

	# Calculate classification accuracy .
	with tf.name_scope('evaluation'):
		correct_predication = tf.equal(tf.argmax(final_tensor, 1),\
					       tf.argmax(ground_truth_input, 1))
		evaluation_step = tf.reduce_mean(tf.cast(correct_predication, tf.float32))

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)

		# Training model for 4000 steps .
		for i in range(STEPS):
			# Input one batch image_data , processed by jpeg_data_tensor 
			# and bottleneck_tensor , then output bottleneck_values and ground_truths 
			# corresponding the input image  as the last full-connected network 
			# training dataset.
			train_bottlenecks, train_ground_truths = \
					get_random_cached_bottlenecks(\
					sess, n_classes, image_lists, BATCH,\
					'training', jpeg_data_tensor, bottleneck_tensor)
			
			sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks,\
							ground_truth_input: train_ground_truths})

			# Get accuracy on validation dataset .
			if i % 100 == 0 or (i + 1) == STEPS:
				# Get one batch of validation data (bottleneck_values and ground_truths)
				# to calculate the model classify accuracy .
				validation_bottlenecks, validation_ground_truths = \
						get_random_cached_bottlenecks(\
						sess, n_classes, image_lists, BATCH,\
						'validation', jpeg_data_tensor, bottleneck_tensor)

				validation_accuracy = sess.run(evaluation_step, \
						feed_dict={bottleneck_input: validation_bottlenecks,
							   ground_truth_input: validation_ground_truths})
				print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' \
						%(i, BATCH, validation_accuracy*100))

		# Get accuracy on test dataset after the model training process finished .
		test_bottlenecks, test_ground_truths = get_test_bottlenecks(\
				sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
		test_accuracy = sess.run(evaluation_step, feed_dict={\
				bottleneck_input: test_bottlenecks,
				ground_truth_input: test_ground_truths})

		print('Final test accuracy = %.1f%%' %(test_accuracy*100))

if __name__ == '__main__':
	tf.app.run()
			















