#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 21:29:21 2018

@author: flyaway
"""
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import pandas as pd
import keras.backend as K

def preprocessing(data):
    # inverse transform one-hot to continuous column
    df_onehot = data[[col for col in data.columns.tolist() if "Soil_Type" in col]]
    #for i in df_onehot.columns.tolist():
    #    if df_onehot[i].sum() == 0:
    #        del df_onehot[i]
    data["Soil"] = df_onehot.dot(np.array(range(df_onehot.columns.size))).astype(int)
    data.drop([col for col in data.columns.tolist() if "Soil_Type" in col], axis = 1, inplace = True)
    label = np.array(OneHotEncoder().fit_transform(data["Cover_Type"].values.reshape(-1, 1)).todense())
    del data["Cover_Type"]
    cate_columns = ["Soil"]
    cont_columns = [col for col in data.columns if col != "Soil"]
    # Feature normilization
    scaler = StandardScaler()
    data_cont = pd.DataFrame(scaler.fit_transform(data[cont_columns]), columns = cont_columns)
    data_cate = data[cate_columns]
    data = pd.concat([data_cate, data_cont], axis = 1)
    cate_values = data[cate_columns].values
    cont_values = data[cont_columns].values
    return data, label, cate_values, cont_values #cate_columns, cont_columns



def embedding_input(inp,name, n_in, n_out):
    #inp = Input(shape = (1, ), dtype = 'int64', name = name)
   #inp = tf.placeholder(tf.int32,shape = (1,))
    print type(inp)
    with tf.variable_scope(name) as scope:
        embeddings = tf.Variable(tf.random_uniform([n_in,n_out],-1.0,1.0))
        #embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
    print embeddings.shape
    return inp,tf.nn.embedding_lookup(embeddings, inp,name = scope.name)


def embedding_feature_generate(cate_values,num_cate):
    #data, label, cate_columns, cont_columns = preprocessing(data)
    embeddings_tensors = []
    
    col = cate_values.shape[1]
    for i in range(col):
        layer_name = 'inp_' + str(i)
        
        #nunique = np.unique(cate_values[:,i]).shape[0]
        nunique = num_cate[i]
        embed_dim = nunique if int(6 * np.power(nunique, 1/4)) > nunique \
            else int(6 * np.power(nunique, 1/4))
        t_inp, t_build = embedding_input(cate_values[:,i],layer_name, nunique, embed_dim)
        embeddings_tensors.append((t_inp, t_build))
        del(t_inp, t_build)
    inp_embed =  [et[1] for et in embeddings_tensors]
    return inp_embed

def fclayer(x,output_dim,reluFlag,name):
    with tf.variable_scope(name) as scope:
        input_dim = x.get_shape().as_list()[1]
        W = tf.Variable(tf.random_normal([input_dim,output_dim], stddev=0.01))
        b = tf.Variable(tf.random_normal([output_dim], stddev=0.01))
        out = tf.nn.xw_plus_b(x,W,b,name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out
        
def crosslayer(x,inp_embed,name):
    with tf.variable_scope(name) as scope:

        input_dim = x.get_shape().as_list()[1]

        w = tf.Variable(tf.random_normal([1, input_dim], stddev=0.01))
        b = tf.Variable(tf.random_normal([1,input_dim], stddev=0.01))
        

        tmp1 = K.batch_dot(K.reshape(x, (-1, input_dim, 1)), tf.reshape(inp_embed,(-1,1,input_dim)))

      
        tmp = K.sum(w * tmp1, 1, keepdims = True)

        tmp = tf.reshape(tmp,shape=(-1,input_dim))

        #one = tf.ones_like(tmp)

        #bb = tf.tensordot(one ,b,1)
        output = tf.add(tmp,b)
        output = tf.add(output,inp_embed)
        return output


def build_model(X_cate,X_cont,num_cate):
        
    inp_embed = embedding_feature_generate(X_cate,num_cate)
    inp_embed = tf.concat([inp_embed],axis = 1,name = "concat")
    input_dim = inp_embed.get_shape().as_list()
    inp_embed = tf.reshape(inp_embed,input_dim[1:3])
    inp_embed = tf.concat([inp_embed,X_cont],axis = 1,name = "concat") 

    fc1 = fclayer(inp_embed,272,reluFlag=False,name = 'fc_1')
    fc2 = fclayer(fc1,272,reluFlag=False,name = 'fc_2')
    fc3 = fclayer(fc2,272,reluFlag=False,name = 'fc_3')
    fc4 = fclayer(fc3,272,reluFlag=False,name = 'fc_4')
    fc5 = fclayer(fc4,272,reluFlag=False,name = 'fc_5')
    fc6 = fclayer(fc5,272,reluFlag=False,name = 'fc_5')
    #return fc6
    #print inp_embed.shape
    cross1 = crosslayer(inp_embed,inp_embed,name = 'cross1')
    #print cross1.shape
    cross2 = crosslayer(cross1,inp_embed,name = 'cross2')
    cross3 = crosslayer(cross2,inp_embed,name = 'cross3')
    cross4 = crosslayer(cross3,inp_embed,name = 'cross4')
    cross5 = crosslayer(cross4,inp_embed,name = 'cross5')
    cross6 = crosslayer(cross5,inp_embed,name = 'cross6')
    cross7 = crosslayer(cross6,inp_embed,name = 'cross7')        
    cross8 = crosslayer(cross7,inp_embed,name = 'cross8') 
    
    output = tf.concat([fc6, cross8], axis = 1,name = 'concat')
    #print output.shape
    out = fclayer(output,7,reluFlag=False,name = 'out')
    return out
    
if __name__ == "__main__":
    data = pd.read_csv("./covtype.csv")
    data, label, cate_values, cont_values = preprocessing(data)
    #获取每种类别特征的最大值，这里仅有一种类别
    num_cate = []
    for i in range(cate_values.shape[1]):   
        nunique = np.unique(cate_values[:,i]).shape[0]
        num_cate.append(nunique)
        
    X_cate = tf.placeholder(tf.int32, [None, cate_values.shape[1]])
    X_cont = tf.placeholder("float",[None,cont_values.shape[1]])
    Y = tf.placeholder("float", [None, 7])



    #print(inp_embed.get_shape().as_list())
    #input_dim = inp_embed.get_shape().as_list()
    #X_train, X_test, y_train, y_test = train_test_split(inp_embed, label, test_size=0.1,random_state=1024)
    output = build_model(X_cate,X_cont,num_cate)
   
    entropy=tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=Y)
    cost = tf.reduce_sum(entropy)
    train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
    predict_op = tf.arg_max(output,1)
    
    ## prepare for data
    X_train_cate, X_test_cate,X_train_cont,X_test_cont,\
    y_train, y_test = train_test_split(cate_values,cont_values,label, test_size=0.1,random_state=1024)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            lost = []
            for start, end in zip(range(0, len(X_train_cate), 256), range(256, len(X_train_cate)+1, 256)):
                sess.run(train_op, feed_dict={X_cate: X_train_cate[start:end], X_cont: X_train_cont[start:end],Y:y_train[start:end]})
                lost.append(sess.run(cost,feed_dict={X_cate: X_train_cate[start:end], X_cont: X_train_cont[start:end],Y:y_train[start:end]}))
            print(i,np.mean(lost))
            print(i,np.mean(np.argmax(y_train, axis=1) == sess.run(predict_op, feed_dict={X_cate: X_train_cate[start:end], X_cont: X_train_cont[start:end]})))
            print(i, np.mean(np.argmax(y_test, axis=1) == sess.run(predict_op, feed_dict={X_cate: X_test_cate[start:end], X_cont: X_test_cont[start:end]})))
