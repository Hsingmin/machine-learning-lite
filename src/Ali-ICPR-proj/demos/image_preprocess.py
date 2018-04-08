
# image_preprocess.py -- Preprocess the Image .

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Adjust image color . Define different order to adjust brightness, constrast,
# hue, saturation and whitening , which may affect the result .
# Programmes can pick one order randomly to reduce the influence on model .
def distort_color(image, color_ordering=0):
	if color_ordering == 0:
		image = tf.image.random_brightness(image, max_delta=32./255.)
		image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
		image = tf.image.random_hue(image, max_delta=0.2)
		image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
	elif color_ordering == 1:
		image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
		image = tf.image.random_brightness(image, max_delta=32./255.)
		image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
		image = tf.image.random_hue(image, max_delta=0.2)
	elif color_ordering == 2:
		image = tf.image.random_hue(image, max_delta=0.2)
		image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
		image = tf.image.random_brightness(image, max_delta=32./255.)
		image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
	else:
		image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
		image = tf.image.random_brightness(image, max_delta=32./255.)
		image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
		image = tf.image.random_hue(image, max_delta=0.2)
	
	# Normalize value in image tensor . 
	return tf.clip_by_value(image, 0.0, 1.0)

# Preprocess given image with size and bounding-box as arguments ,
# converse original image for trianing to neural network input .
#
# Preprocess image steps :
# 	tf.image.decode_jpeg()
#	tf.image.resize_images()
# 	tf.sample_distorted_bounding_box()
#	tf.image.convert_image_dtype()
#	tf.expand_dims()
#	tf.image.draw_bounding_boxes()
#	tf.reshape()
#	tf.slice()
def preprocess_for_train(image, height, width, bbox):
	# Regard whole image as the attention part if bbox is none .
	if bbox is None:
		bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1,1,4])
		
	# Reshape image size at first .
	image = tf.image.resize_images(image,\
			[height, width], method=np.random.randint(4))
	
	# Distort image randomly to reduce affect to model of noise .
	bbox_begin, bbox_size, draw_bbox  = tf.image.sample_distorted_bounding_box(tf.shape(image), 
			bounding_boxes=bbox, min_object_covered=0.1)

	# Convert image tensor data type .
	#
	# Image data type from uint8 to tf.float32 .
	# Expand dimensions from 3-D to 4-D .
	
	if image.dtype != tf.float32:
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)	
	
	image = tf.expand_dims(image, 0)
	
	# tf.image.draw_bounding_boxes(arg1=image, arg2=draw_box)
	#
	# image : dimendion expanded 
	# draw_box : the 3rd result returned by tf.image.sample_distorted_bounding_box()
	distorted_image = tf.image.draw_bounding_boxes(image, draw_bbox)	
	
	distorted_image = tf.reshape(distorted_image, [height, width, 3])
	
	distorted_image = tf.slice(distorted_image, bbox_begin, bbox_size)
	
	distorted_image = tf.image.random_flip_left_right(distorted_image)
	distorted_image = distort_color(distorted_image, np.random.randint(5))

	return distorted_image

# Get image raw data in bytes type .
image_raw_data = tf.gfile.GFile("./to/picture.jpg", "rb").read()

with tf.Session() as sess:
	img_data = tf.image.decode_jpeg(image_raw_data)
	boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.4, 0.5, 0.6]]])

	for i in range(6):
		# Resize image to 299*299
		result = preprocess_for_train(img_data, 299, 299, boxes)
		plt.imshow(result.eval())
		plt.show()
	
































