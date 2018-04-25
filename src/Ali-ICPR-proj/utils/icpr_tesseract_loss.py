# -*- coding: utf-8 -*-
#
# -*- version:
#		python 3.5.2
#		tensorflow 1.4.0
#		numpy 1.13.1
#		tesseract 4.0
# -*- author: Hsingmin Lee
#
# icpr_tesseract_accuracy.py -- Calculate accuracy of tesseract tool 
# to identify text on sliced image.

import os
import sys
import codecs

def calculate_match_rate(text_a, text_b):
	cnt = 0
	for w in text_a:
		if w in text_b:
			cnt += 1
	return float(cnt/(len(text_a)+1))

def calculate_tesseract_accuracy():
	whole_match_count = 0
	single_match_count = 0
	text_count = 0

	with codecs.open("./train_logits.txt", 'r', 'utf-8') as file:

		for line in file:
			text_count += 1
			line = line.split(':')
			if len(line) == 1:
				continue
			line[1] = line[1].strip().replace(' ', '')
			print(line)
			if line[0] == line[1]:
				whole_match_count += 1
			single_match_count += calculate_match_rate(line[0], line[1])
		text_count += 1

	print('whole text match rate = ', float(whole_match_count/text_count))
	print('text distance match rate = ', float(single_match_count/text_count))

if __name__ == '__main__':

	# whole text match rate = 0.07
	# text distance match rate = 0.15
	calculate_tesseract_accuracy()


























