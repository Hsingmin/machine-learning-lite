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
# gbmc.py -- Gradient Boosting Machine Classifier.
#
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Classifier or predictor based on LightGBM.
# params:
#       train        -- training dataset
#       validate     -- validation dataset
#       test         -- test dataset
#       is_train     -- model for training or predicting
#       best_iter    -- model with minimum log_loss on validation dataset
#                       default as 20000 when training 
# returns:
#       best_iter    -- training model with minimum log_loss on best_iter
#       None         -- save result into csv file
#
def lgbmc(train=None, validate=None, test=None, is_train=True, iteration=20000):

    non_critical_features = ['is_trade', 'item_category_list', 'item_property_list',
                             'predict_category_property', 'instance_id', 'context_id',
                             'realtime', 'context_timestamp']
    col = [c for c in train if c not in non_critical_features]
    X_train = train[col]
    y_train = train['is_trade'].values
    # Produce validate dataset for model training process
    if is_train == True:
        X_validate = validate[col]
        y_validate = validate['is_trade'].values

        print("Training lgbmc Model Start  ........................ ")
    if is_train == False:
        print("Lgbmc Model Predict for Test Dataset ................")

    # params:
    #       boosting_type           -- algorithm for gradient boosting model
    #       objective               -- task type 'regression'/'binary'/'multiclass' 
    #       num_leaves              -- max leaves for base learner
    #       max_depth               -- max depth for base learner
    #       learning_rate           -- boosting learning rate
    #       colsample_bytree        -- subsample ratio of columns when constructing each tree
    #       subsample               -- subsample ratio of training distance
    #       min_sum_hessian_in_leaf -- minimal sum hessian in one leaf used to deal with over-fitting
    #       n_estimators            -- number of boosted trees to fit
    lgbm_classifier = lgb.LGBMClassifier(boosting_type='gbdt',
                                         objective='binary',
                                         num_leaves=35,
                                         max_depth=8,
                                         learning_rate=0.03,
                                         seed=2018,
                                         colsample_bytree=0.8,
                                         subsample=0.9,
                                         min_sum_hessian_in_leaf=100,
                                         n_estimators=iteration)
    # params:
    #       X_train                -- input feature matrix
    #       y_train                -- input label matrix
    #       eval_set               -- validate data in tuple type
    #       early_stopping_rounds  -- training rounds for evaluating validate error to activate early-stopping
    # returns:
    #       self object
    #
    if is_train == True:
        lgbm_model = lgbm_classifier.fit(X_train, y_train,
                                         eval_set=[(X_validate, y_validate)],
                                         early_stopping_rounds=200)
        # class attribute best_iteration_ is the best iteration of fitted model 
        # when early_stopping_rounds parameter specified.   
        best_iter = lgbm_model.best_iteration_
    else:
        lgbm_model = lgbm_classifier.fit(X_train, y_train)

    # Array including all numerical features.
    predictors = [c for c in X_train.columns]
    # Array including feature importances. 
    feature_importance = pd.Series(lgbm_model.feature_importances_, predictors).sort_values(ascending=False)
    print("Output Feature Importance Series as : ")
    print("======================================")
    print(feature_importance)

    # Class method predict_proba(X, raw_score=False, num_iteration=0)
    # params:
    #       X                     -- input feature matrix 
    # returns:
    #       predicted_probability -- predicted probability for each class for each sample
    #                                in shape of [n_samples, n_classes]
    if is_train == True:
        predicted_prob = lgbm_model.predict_proba(validate[col])[:, 1]
        validate['predict'] = predicted_prob
        validate['index'] = range(len(validate))
        print("Evaluate Model : ")
        validate['log_loss'] = log_loss(validate['is_trade'], validate['predict'])
        validate[['index', 'predict', 'log_loss']].to_csv('./to/train_validate_loss.txt', sep=" ", index=False)
        print('Logistic Loss = ', validate['log_loss'])
        print("The Best Iteration is : ", best_iter)
        return best_iter
    else:
        # predicted_prob = lgbm_model.predict_proba(test[col])[:, 1]
        predicted_probability = lgbm_model.predict_proba(test[col])
        test['is_trade_probability'] = predicted_probability[:, 1]
        test['not_trade_probability'] = predicted_probability[:, 0]
        # test['predicted_score'] = predicted_prob
        # res = test[['instance_id', 'predicted_score']]
        res = test[['instance_id', 'is_trade_probability', 'not_trade_probability']]
        return res


