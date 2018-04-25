# Ali-ICPR MTWI 2018挑战赛：网络图片文字识别

1. train.py
    以预处理程序得到的slice作为训练集，以ocr.model作为模型进行训练；
    BATCH_SIZE = 32
    图片统一进行二值化，高度height=32，Batch内图片宽度对齐，便于神经网络的输入；
    图片名称作为Label，Label长度也要进行对齐；
    OCR model Input:
    #   input = Input(name='the_input', shape=(height, None, 1))
    #   labels = Input(name='the_labels', shape=[None,], dtype='float32')
    #   input_length = Input(name='input_length', shape=[1], dtype='int64')
    #   label_length = Input(name='label_length', shape=[1], dtype='int64')
    
    实际输入形式为：
    X, Y = [X_batch, y_batch, np.ones(batch_size)*input_length,
            np.ones(batch_size)*label_length], np.ones(batch_size)

2. 使用Batch Normalization Layer模型收敛速度较快：
    0 steps: loss = 622.68
    ...
    40 steps: loss = 75.21
    ...
    笔记本性能较弱，使用GPU训练试试；

Version：
  OS：Windows7
  python：3.5.2
  numpy: 1.13.1
  tensorflow: 1.4.0
  keras: 2.1.2
  
联系我：hsingmin.lee@yahoo.com

