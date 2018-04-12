# -*- coding: utf-8 -*-
#
# -*- version:
#		python 3.5.2
#		tensorflow 1.4.0
#		numpy 1.13.1
#		tesseract 4.0
# -*- author: Hsingmin Lee
#
# icpr_image_tesseract.py -- Recognize text on image 
# sliced by icpr_image_slice.py .

import os
import sys
import codecs

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
	
import pyocr
from PIL import Image

# Train slices store path
TRAIN_SLICES_DIR = 'd:/engineering-data/Ali-ICPR-data/train_slice'

TRAIN_BINARY_DIR = 'd:/engineering-data/Ali-ICPR-data/train_binary'

DEMOS_SLICES_DIR = './demos/to'

DEMOS_RESULT_DIR = './demos/to/demos_logits.txt'

TESSERACT_RESULT_DIR = 'd:/engineering-data/Ali-ICPR-data/train_logits.txt'

tools = pyocr.get_available_tools()[:]
if(0 == len(tools)):
	print('No usable tools, please check the tesseract install environment.')
	exit(1)

def get_slice_list(path):
	slice_list = []
	label_list = []

	for rootdir, subdirs, filenames in os.walk(path):
		for filename in filenames:
			slice_list.append(os.path.join(rootdir, filename))
			label_list.append(filename.replace('.jpg', '').replace('.png', ''))

	return slice_list, label_list

def produce_binary_image(path):
	images, labels = get_slice_list(DEMOS_SLICES_DIR)
	bimages = []
	for i in range(len(images)):
		img = Image.open(images[i]).convert("L")
		bimage = os.path.join(path, labels[i] + '.png')
		img.save(bimage)
		bimages.append(bimage)
	
	return bimages, labels

def image_tesseract():
	
	images, labels = produce_binary_image(TRAIN_BINARY_DIR)

	with codecs.open(TESSERACT_RESULT_DIR, 'w', 'utf-8') as file:

		for i in range(len(images)):
			'''
			print('image : ', images[i], ' correspond to ', labels[i])
			print('text recognize: ')
			print(tools[0].image_to_string(Image.open(images[i]),
			lang='chi_sim'))
			print('==============================')
			'''
			try:
				logit = tools[0].image_to_string(Image.open(images[i]), lang='chi_sim')
				line = labels[i] + ':' + logit + '\r\n'
				file.write(line)
			except Exception as e:
				print(e)
				pass

if __name__ == '__main__':
	image_tesseract()


























