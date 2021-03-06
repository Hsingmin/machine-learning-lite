# machine-learning-lite
A quick and light check list for machine learning learning.

2018年春节假期，对2017年9月以来学习的Machine Learning知识点进行简要的总结，随手记录下来；
机器学习部分主要集中在第7章，参考学习资料包含周志华《机器学习》，李航《统计学习方法》，《PRML》，《Deep Learning》，Tensorflow Docs，
Python Docs以一些技术博客；
其他章节来源于仓库 https://github.com/taizilongxu/interview_python#1-python
对Python部分进行了整理和充实；
C/C++，Algorithm，OS，网络和数据库章节尚待完成；
本书初稿使用MS-word2010编写，公式使用内嵌工具编辑；
Python部分代码均调试通过：
  OS：Windows7
  python：3.5.2
  numpy: 1.13.1
  tensorflow: 1.4.0
  
联系我：hsingmin.lee@yahoo.com

目录
1.	C/C++	6
1.1	const关键字	6
1.2函数传值与传引用	11
1.3 字符串转整型问题	12
1.4 操作符重载	12
2.	Python	13
2.1	可变类型与非可变类型	13
2.2	静态方法，实例方法与类方法	13
2.3	类变量与实例变量	14
2.4	Python自省特性	16
2.5	列表推导式与字典推导式	16
2.6	深拷贝与浅拷贝	17
2.7	类方法__new__( )与__init__( )	19
2.8	字符串格式化:%与.format	19
2.9	Python的*args和**kwargs	20
2.10	常见的几种设计模式	21
2.11	Python变量的作用域	22
2.12	GIL线程全局锁及协程	23
2.13	闭包	23
2.14	Lambda与函数式表达式	24
2.15	编码与解码(incode/decode)	24
2.16	迭代器与生成器	25
2.17	装饰器	26
2.18	Python中的重载	27
2.19	Python新式类与旧式类	28
2.20	邮箱地址正则表达式	29
2.21	Python内置类型list/dictionary/tuple/string	33
2.22	Python中的is	35
2.23	read/readline和readlines	35
2.24	垃圾回收	35
3.	Algorithm	35
3.1	时间复杂度计算	35
3.2	二叉树	36
3.2.1 二叉树的数据结构	36
3.2.2 二叉树的生成	37
3.2.3 二叉树的遍历算法	37
3.3	最大堆	42
3.4	红黑树	42
3.5	B树	42
3.6	线性结构-栈与队列	42
3.7	线性结构-链表	42
3.8	寻找链表倒数第k个节点	42
3.9	快速排序	43
3.10	堆排序	43
3.11	无序数字列表寻找所有间隔为d的组合	43
3.12	列表[a1, a2, a3, …, an]求其所有组合	44
3.13	一行python代码实现1+2+3+…+10**8	44
3.14	长度未知的单向链表求其是否有环	44
3.15	单向链表应用快速排序	44
3.16	长度为n的无序数字元素列表求其中位数	44
3.17	遍历一个内部未知的文件夹	44
3.18	台阶问题/斐波那契	44
3.19	变态台阶问题	44
3.20	矩形覆盖问题	45
4.	OS	46
4.1	多线程与多进程的区别	46
4.2	协程	46
4.3	进程间通信方式	46
4.4	虚拟存储系统中缺页计算	46
4.5	并发进程不会引起死锁的资源数量计算	46
4.6	常用的Linux/git命令和作用	46
4.7	查看当前进程的命令	46
4.8	46
5.	网络	47
5.1	TCP/IP协议	47
5.2	OSI五层协议	47
5.3	Socket长连接	47
5.4	Select与epoll	47
5.5	TCP与UDP协议的区别	47
5.6	TIME_WAIT过多的原因	47
5.7	http一次连接的全过程描述	47
5.8	http连接方式——get与post的区别	47
5.9	restful	48
5.10	http请求的状态码 200/403/404/504	48
6.	数据库	49
6.1	MySQL锁的种类	49
6.2	死锁的产生	49
6.3	MySQL的char/varchar/text的区别	49
6.4	Join的种类与区别	49
6.5	A LEFT JOIN B的查询结果中，B缺少的部分如何显示	49
6.6	索引类型的种类	49
6.7	BTree索引与hash索引的区别	49
6.8	如何对查询命令进行优化	49
6.9	NoSQL与关系型数据库的区别	49
6.10	Redis常用的存储类型	50
7.	机器学习	51
7.1	模型评估方法	51
7.1.1	数据集划分	51
7.1.2	模型的性能度量	51
7.1.3	偏差-误差分解	51
7.2	LR与极大似然估计	52
7.3	LDA	53
7.4	类别不平衡问题	53
7.5	判别/生成模型与先验/后验概率	53
7.6	朴素贝叶斯分类器	54
7.7	常用的损失函数	55
7.7.1	损失函数的数学解释	55
7.7.2	Tensorflow中的损失函数	55
7.8	决策树	56
7.9	连续值与缺失值处理	57
7.10	BP算法	57
7.11	无约束最优化问题求解	59
7.11.1	拉格朗日乘子法与KKT条件	59
7.11.2	梯度下降法	59
7.11.3	牛顿法/拟牛顿法	60
7.12	最大熵模型与IIS	61
7.12.1	最大熵模型	61
7.12.2	IIS	63
7.12.3	熵/条件熵/互信息	64
7.13	局部最小与全局最小	64
7.14	支持向量机与拉格朗日乘子法	65
7.15	正则化	68
7.15.1	LASSO与Ridge	68
7.15.2	L1正则化求解	69
7.16	EM算法	70
7.17	Boosting与Bagging	70
7.17.1	Boosting方法	71
7.17.2	Bagging方法	74
7.18	监督学习与无监督学习	75
7.19	聚类	75
7.20	特征选择与稀疏学习	77
7.20.1	维度灾难	77
7.20.2	特征选择	77
7.21	PCA	78
7.21.1	特征值分解	78
7.21.2	奇异值分解	79
7.21.3	主成分分析	80
7.22	Apriori算法与FP-Growth	81
7.22.1	关联规则	81
7.22.2	Apriori算法	81
7.22.3	FP-Growth算法	82
7.23	词频-逆文本词频	83
7.24	Page-Rank	83
7.24.1	爬虫与倒排索引	83
7.24.2	PR迭代更新	84
7.24.3	话题倾向排序	84
7.25	概率图模型	85
7.25.1	隐马尔科夫模型	85
7.25.2	马尔科夫随机场	88
7.25.3	条件随机场	89
7.25.4	隐狄利克雷分配模型	93
7.26	从伯努利到狄利克雷分布	95
7.26.1	伯努利分布	95
7.26.2	二项分布	95
7.26.3	Beta分布	96
7.26.4	多项式分布	96
7.26.5	Dirichlet分布	97
7.27	常用的不等式	98
8.	深度学习	98
8.1	神经网络基本框架	98
8.1.1	神经网络训练	98
8.1.2	神经网络优化	99
8.2	全连接神经网络	101
8.2.1	激活函数	101
8.2.2	前向传播	101
8.3	卷积神经网络	102
8.3.1	常用的图像数据集	102
8.3.2	CNN	102
8.3.3	LeNet-5	103
8.3.4	迁移学习	105
8.4	循环神经网络	107
8.4.1	RNN	107
8.4.2	LSTM	108
8.4.3	GRU	109
8.4.4	梯度消散/爆炸与序列截断	109
8.4.5	BPTT	110
8.4.6	CNN与RNN对比	111
8.4.7	NLP	111
8.4.8	Word2vec	115
8.4.9	TF-Learn	116
8.5	RBM与吉布斯采样	117
8.5.1	受限玻尔兹曼机	117
8.5.2	吉布斯采样	119
8.6	GAN	119
8.7	Tensorflow	120
8.7.1	三大组件	120
8.7.2	Collection	121
8.7.3	Variable和变量管理	121
8.7.4	多线程	122
8.7.5	数据预处理	124
8.7.6	图像识别通用框架	128
8.7.7	模型持久化	129
8.7.8	常用生成函数	131
8.7.9	常用数据处理函数	131
8.7.10	常用随机数生成函数	132
8.7.11	常用文件处理函数	132


