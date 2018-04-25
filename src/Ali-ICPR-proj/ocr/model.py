# -*- coding: utf-8 -*-
# -*- version:
#   python: 3.5.2
#   tensorflow: 1.4.0
#   keras: 2.1.2
#   numpy: 1.14.0
# package: ocr
# author: Hsingmin Lee
# ocr.model defines the CNN+GRU+CTC structure to solve text reconize task
# for Ali-ICPR-2018 match.

from keras.layers import Input,Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers import Flatten,BatchNormalization,Permute,TimeDistributed,Dense,Bidirectional,GRU
from keras.models import Model
from keras.layers import Lambda
from keras.optimizers import SGD
import numpy as np
import keras.backend  as K
import ocr.keys as keys
import os

# Get CTC loss on each batch element.
# Using Tensorflow backend to process underlying operation.
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_model(height,nclass):
    rnnunit  = 256
    # Input binary image shape=[height, width, channel].
    input = Input(shape=(height,None,1),name='the_input')
    m = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same',name='conv1')(input)
    m = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool1')(m)
    m = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same',name='conv2')(m)
    m = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool2')(m)
    m = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same',name='conv3')(m)
    m = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same',name='conv4')(m)

    # Zeros padding for image cols, and no padding for rows. 
    m = ZeroPadding2D(padding=(0,1))(m)
    m = MaxPooling2D(pool_size=(2,2),strides=(2,1),padding='valid',name='pool3')(m)

    m = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same',name='conv5')(m)
    # Batch normalization.
    m = BatchNormalization(axis=1)(m)
    m = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same',name='conv6')(m)
    m = BatchNormalization(axis=1)(m)
    m = ZeroPadding2D(padding=(0,1))(m)
    m = MaxPooling2D(pool_size=(2,2),strides=(2,1),padding='valid',name='pool4')(m)
    m = Conv2D(512,kernel_size=(2,2),activation='relu',padding='valid',name='conv7')(m)

    # Rearrange the dimensions of input tensor to output, 
    # often used for connection between CNN and RNN.
    m = Permute((2,1,3),name='permute')(m)
    # image permutated to [width, height, channel], in which width will
    # be regarded as time step dimension.
    m = TimeDistributed(Flatten(),name='timedistrib')(m)

    # Get the whole sequence of Bidirectional GRU.
    m = Bidirectional(GRU(rnnunit,return_sequences=True),name='blstm1')(m)
    m = Dense(rnnunit,name='blstm1_out',activation='linear')(m)
    m = Bidirectional(GRU(rnnunit,return_sequences=True),name='blstm2')(m)
    y_pred = Dense(nclass,name='blstm2_out',activation='softmax')(m)

    # Create CNN+BiGRU model.
    basemodel = Model(inputs=input,outputs=y_pred)

    # CTC model with [input, labels, input_length, label_length] as inputs,
    # [loss_out] as outputs.
    labels = Input(name='the_labels', shape=[None,], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[input, labels, input_length, label_length], outputs=[loss_out])
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    #model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')

    # Model must be compiled before used, or it would raise exception when fit and evaluated.
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    #model.summary()
    return model,basemodel

def load_model(height, nclass, basemodel):
    # Characters table in keys.py
    characters = keys.alphabet[:]

    # Persistent model weights loaded in.
    modelPath = os.path.join(os.getcwd(),"ocr0.2.h5")
    height = 32
    nclass = len(characters)
    if os.path.exists(modelPath):
        # model,basemodel = get_model(height,nclass+1)
        basemodel.load_weights(modelPath)

    return basemodel

def predict(im, basemodel):
    # Convert image into binary format.
    im = im.convert('L')
    # Scalling image.
    scale = im.size[1]*1.0 / 32
    w = im.size[0] / scale
    w = int(w)
    im = im.resize((w,32))
    # Convert to 256-grayscale image.
    img = np.array(im).astype(np.float32)/255.0
    X  = img.reshape((32,w,1))
    # X = img.reshape((1, 32, w, 1))
    X = np.array([X])
    # Model method with batch data as input to get predict result.    
    y_pred = basemodel.predict(X)
    y_pred = y_pred[:,2:,:]
    # String object decode.
    out    = decode(y_pred)
    # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )[0][0])[:, :]
    # out = u''.join([characters[x] for x in out[0]])

    if len(out)>0:
        while out[0]==u'。':
            if len(out)>1:
               out = out[1:]
            else:
                break

    return out

def decode(pred):
        # Characters table in keys.py
        characters = keys.alphabet[:]
        charactersS = characters+u' '
        # Get max value index in axis 2.
        t = pred.argmax(axis=2)[0]
        length = len(t)
        char_list = []
        n = len(characters)
        for i in range(length):
            if t[i] != n and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(charactersS[t[i] ])
        return u''.join(char_list)
