# -*- coding: utf-8 -*-
#
# -*- version:
#		python 3.5.2
#		tensorflow 1.4.0 
#		numpy 1.13.1
# -*- author: Hsingmin Lee
#
# dataset.py -- import data.dataset to get train/validate-dataset and
# test dataset .

import os
import numpy as np
import tensorflow as tf

import sys
sys.path.append(r"D:\python_work\machine-learning\engineering\Ali-ICPR-proj")
import ocr.model as om

characters = om.keys.alphabet[:]

class Dataset(object):
    def __init__(self, data_path, validation_percentage, test_percentage):
        self.data_path = data_path
        self.validation_percentage = validation_percentage
        self.test_percentage = test_percentage
        self.test = []
        self.validation = []
        self.train = []

    def split(self):
        img_list = []
        for rootdir, subdirs, filenames in os.walk(self.data_path):
            for filename in filenames:
                img_list.append(os.path.join(rootdir, filename))

        for img in img_list:
            chance = np.random.randint(100)
            if chance < self.test_percentage:
                self.test.append(img)
            elif chance < self.test_percentage + self.validation_percentage:
                self.validation.append(img)
            else:
                self.train.append(img)

    def get_train(self):
        return self.train
    def get_test(self):
        return self.test
    def get_validation(self):
        return self.validation

class AlignedBatch(object):
    def __init__(self, height=32, width=256):
        self.height = height
        self.width = width

    def __call__(self, batch):
        aligned_batch = []
        self.width = max(np.shape(sample)[1] for sample in batch)
        for sample in batch:
            padding = np.array([[0]*(self.width-np.shape(sample)[1])]*np.shape(sample)[0])
            padding_sample = np.concatenate((sample, padding), axis=1)
            aligned_batch.append(padding_sample)
        return aligned_batch

class AlignedOnehot(object):
    def __init__(self, length=10, characters=characters):
        self.length = length
        self.characters = characters

    def __call__(self, batch):
        self.length = max(len(sample) for sample in batch)
        label = [np.zeros(self.length)]*len(batch)

        for j, sample in enumerate(batch):
            if len(sample) < self.length:
                sample += u' ' * (self.length-len(sample))

            for i, char in enumerate(sample):
                index = self.characters.find(char)
                if index == -1:
                    index = self.characters.find(u' ')
                label[j][i] = index
        return label














