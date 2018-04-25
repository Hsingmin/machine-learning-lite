# -*- coding: utf-8 -*-
#
# -*- version:
#		python 3.5.2
#		tensorflow 1.4.0
#		numpy 1.13.1
#		tesseract 4.0
# -*- author: Hsingmin Lee
#
# icpr_image_slice.py -- Draw bounding boxes on images in dataset
# provided by Ali-ICPR MTWI 2018.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import codecs

# Train images store path
TRAIN_IMAGE_DIR = 'd:/engineering-data/Ali-ICPR-data/train_image_9000'
# Train images bounding box path
TRAIN_BBOX_DIR = 'd:/engineering-data/Ali-ICPR-data/train_txt_9000'
# Train slices store path
TRAIN_SLICES_DIR = 'd:/engineering-data/Ali-ICPR-data/train_slice/'

# Validate images store path
VALIDATE_IMAGE_DIR = 'd:/engineering-data/Ali-ICPR-data/validate_image_1000'
# Validate images bounding box path
VALIDATE_BBOX_DIR = 'd:/engineering-data/Ali-ICPR-data/validate_txt_1000'
# Validate slices store path
VALIDATE_SLICES_DIR = 'd:/engineering-data/Ali-ICPR-data/validate_slice/'

# Preprocess image with given bounding-box as arguments, and get image slices .
#
# Preprocess image steps :
# 	create_images_list()
#	get_single_image_slice()
# 	get_single_image_bboxes()

# Get all images list and bounding box list
# arguments:
#	None
# returns:
#	image_list: list of full path for all images in training dataset
#	bbox_list: list of full path for all bounding boxes in training dataset
def create_image_list():
	image_list = []
	bbox_list = []
	
	for rootdir, subdirs, filenames in os.walk(TRAIN_IMAGE_DIR):
		for filename in filenames:
			image_list.append(os.path.join(rootdir, filename))
	
	for rootdir, subdirs, filenames in os.walk(TRAIN_BBOX_DIR):
		for filename in filenames:
			bbox_list.append(os.path.join(rootdir, filename))

	return image_list, bbox_list

# Get single image bounding boxes 
# arguments: 
#	bbox_dir: bounding box file stored directory
# returns:
#	bboxes: a list of single image bounding boxes 
#	labels: a list of bounding box labels
def get_single_image_bboxes(bbox_dir):

	bbox_list = []
	labels = []
	with codecs.open(bbox_dir, 'r', 'utf-8') as bf:
		for line in bf:
			box = line.strip().split(',')
			labels.append(box[-1])
			bbox_list.append(box[:-1])

	'''
	sized_image = np.asarray(image_data.eval(session=sess), dtype='uint8')
	height = len(sized_image)
	width = len(sized_image[0])
	'''
	bboxes = []
	for bbox in bbox_list:
		bboxes.append([int(float(bbox[1])), int(float(bbox[0])), 
			int(float(bbox[5])), int(float(bbox[4]))])
	
	return bboxes, labels

def is_bbox_invalid(sess, width, height, bbox):
	
	# print("current image width = %d, height = %d" %(width,height))
	if bbox[2]-bbox[0] < 5 or bbox[3]-bbox[1] < 5:
		return True
	elif bbox[2]-bbox[0] > (width-10) or bbox[3]-bbox[1] > (height-10):
		return True
	elif bbox[0] * bbox[1] * bbox[2] * bbox[3] < 0:
		return True
	elif bbox[1] > height or bbox[3] > height or bbox[0] > width or bbox[2] > width:
		return True
	else:
		return False

def is_label_invalid(label):
	invalid_characters = {'\\', '/', ':', '?', '*', '"', '<', '>', '|'}
	if label == '###':
		return True
	for s in label:
		if s in invalid_characters:
			return True
	return False

def get_soft_margin(begin, size, width, height):
	if begin[0]-10 > 0:
		begin[0] = begin[0]-10
	if begin[1]-10 > 0:
		begin[1] = begin[1]-10
	if size[0]+10 < width:
		size[0] = size[0]+10
	if size[1]+10 < height:
		size[1] = size[1]+10
	return begin, size
	

# Get single image slice
# arguments:
#	image: a single decoded image data 
#	bboxes: bounding boxes list for single image
#	labels: labels corresponding to bbox in bboxes
# returns:
#	None
def get_single_image_slice(sess, image, bboxes, labels):
	for i in range(len(bboxes)):
		try:
			bbox = bboxes[i]
			label = (labels[i]).replace('/', '').strip()
			# illegal character handler
			sized_image = np.asarray(image.eval(session=sess), dtype='uint8')
			height = len(sized_image)
			width = len(sized_image[0])
			if is_label_invalid(label) or is_bbox_invalid(sess, width, height, bbox):
				continue
		
			path = TRAIN_SLICES_DIR + label + ".png"
			begin, size = get_soft_margin([bbox[0], bbox[1], 0], [bbox[2]-bbox[0], bbox[3]-bbox[1], -1], width, height)
			sliced_image = tf.slice(image, begin, size)
			#reshaped_image = tf.image.resize_images(sliced_image, [300, 300], method=0)
			reshaped_image = sliced_image
			uint8_image = tf.image.convert_image_dtype(reshaped_image, dtype=tf.uint8)
			encoded_image = tf.image.encode_png(uint8_image)
			with tf.gfile.GFile(path, "wb") as f:
				f.write(encoded_image.eval())
		except Exception as e:
			print(e)
			pass

def main(argv=None):
	# Get images list and corresponding boxes list.
	image_list, bbox_list = create_image_list()
	with tf.Session() as sess:
		for i in range(len(image_list)):
			image_dir = image_list[i]
			bbox_dir = bbox_list[i]
			print(image_dir)
			print("Corresponding to : ")
			print(bbox_dir)
			print("============================")
			image_raw_data = tf.gfile.FastGFile(image_dir, "rb").read()
			image_data = tf.image.decode_jpeg(image_raw_data)
			bboxes, labels = get_single_image_bboxes(bbox_dir)
			get_single_image_slice(sess, image_data, bboxes, labels)


if __name__ == '__main__':
	main(argv=None)































