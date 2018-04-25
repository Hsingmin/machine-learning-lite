# Ali-ICPR MTWI 2018挑战赛：网络图片文字识别

1. dataset.py
    数据预处理文件，提供了三个类：

    Dataset:
        划分数据集，得到训练集train(80%), 验证集validation(10%), 测试集test(10%)；

    AlignedBatch：
        图片样本 X_batch对齐，height=32， width取同一batch内最长的width，缺少的部分补0；

    AlignedOnehot：
        图片标签 y_batch对齐，length取同一batch内样本长度最大值，缺少部分补u' '；

Version：
  OS：Windows7
  python：3.5.2
  numpy: 1.13.1
  tensorflow: 1.4.0
  keras: 2.1.2
  
联系我：hsingmin.lee@yahoo.com

