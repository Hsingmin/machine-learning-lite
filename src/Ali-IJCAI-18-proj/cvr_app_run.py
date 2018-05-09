# -*- coding: utf-8 -*-
#
# -*- version:
#		python 3.5.2
#		numpy 1.13.1
#		pandas 0.20.3
#		sklearn 0.19.0
#
# -*- author: Hsingmin Lee
#
# cvr_ffm.py -- Create FFM model to get CVR prediction .
#
import os
import sys
import codecs
import numpy as np
import pandas as pd
import csv
import data.dataset as dt
import model.gbmc as gc

# Train slices store path
DATASET_DIR = 'd:/engineering-data/Ali-IJCAI-18-data'

TRAIN_DATASET_RAW = 'round1_ijcai_18_train_20180301.txt'

TEST_DATASET_RAW = 'round1_ijcai_18_test_a_20180301.txt'

# Feature encode and combine on raw dataset.
# params:
#       dataset -- TRAIN_DATASET_RAW or TEST_DATASET_RAW
# returns:
#       dataset -- dataframe with encoded and combined features
#
def feature_encode_combine(dataset):
    dataset = dataset.drop_duplicates(subset='instance_id')
    dataset = dt.feature_process(dataset)
    dataset = dt.time_hour_encode(dataset)
    dataset = dt.split_shop_feature(dataset)
    # dataset = dt.date_mask(dataset)
    dataset = dt.feature_dtype_convert(dataset)
    dataset = dt.item_feature_internal_combine(dataset)
    dataset = dt.user_feature_internal_combine(dataset)
    dataset = dt.user_item_feature_combine(dataset)
    dataset = dt.user_shop_feature_combine(dataset)
    dataset = dt.shop_item_feature_combine(dataset)

    return dataset

# Undersample train dataset to make balance between positive and
# negative samples.
# params:
#       train -- train dataset with features encoded and combined
# returns:
#       dataset -- dataframe with balanced samples
#
def train_undersampling(train):
    trade_series = train.pop('is_trade')
    train.insert(len(train.columns), 'is_trade', trade_series)

    # Undersampling train dataset to make positive and negative samples balanced.
    positive_samples = train[train['is_trade'] == 1]
    negative_samples = train[train['is_trade'] == 0]
    # Reset index of negative_samples for sampling operation.
    negative_samples = negative_samples.reset_index(drop=True)
    sampled_negative = pd.DataFrame()
    pm = len(positive_samples)
    nm = len(negative_samples)
    for i in range(pm):
        sampled_negative = sampled_negative.append(negative_samples.loc[np.random.randint(nm)])
    dataset = pd.concat([positive_samples, sampled_negative])
    # Shuffle dataset.
    dataset = dataset.sample(frac=1)

    return dataset

# Run GBDT model on training dataset.
# params:
#       train -- dataset need furtherly splited into training and validation dataset
#       test -- evaluate dataset
# output:
#       best_iter -- number of child trees of GBDT model with minimum log loss
#       predicts.txt -- predicted result of test dataset
#
def lgbmc_run(train, test):

    train = train.loc[0: int(len(train)*0.8)]
    validate = train.loc[int(len(train)*0.8): len(train)]
    best_iter = gc.lgbmc(train=train, validate=validate, is_train=True, iteration=20000)

    # Predict on test dataset
    predicts = gc.lgbmc(train=train, test=test, is_train=False, iteration=best_iter)
    predicts.to_csv('./to/predict_result.txt', sep=" ", index=False)

    return best_iter

def ffm_run():

    return 0

if __name__ == '__main__':

    if not os.path.exists(os.path.join(DATASET_DIR, 'trainable_data.txt')):
        train = pd.read_csv(os.path.join(DATASET_DIR, TRAIN_DATASET_RAW), sep="\s+")
        train = feature_encode_combine(train)
        train = train_undersampling(train)
        train.to_csv(os.path.join(DATASET_DIR, 'trainable_data.txt'), sep=" ", index=False)
    else:
        print("trainable_data already existed ......")
        train = pd.read_csv(os.path.join(DATASET_DIR, 'trainable_data.txt'), sep="\s+")
        # train.to_csv(os.path.join(DATASET_DIR, 'trainable_data.csv'))

    if not os.path.exists(os.path.join(DATASET_DIR, 'predictable_data.txt')):
        test = pd.read_csv(os.path.join(DATASET_DIR, TEST_DATASET_RAW), sep="\s+")
        test = feature_encode_combine(test)
        test.to_csv(os.path.join(DATASET_DIR, 'predictable_data.txt'), sep=" ", index=False)
    else:
        print("predictable_data already existed ......")
        test = pd.read_csv(os.path.join(DATASET_DIR, 'predictable_data.txt'), sep="\s+")
        # test.to_csv(os.path.join(DATASET_DIR, 'predictable_data.csv'))

    best_iter = lgbmc_run(train, test)












