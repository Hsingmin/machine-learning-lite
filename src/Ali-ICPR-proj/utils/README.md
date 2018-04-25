# Ali-ICPR MTWI 2018挑战赛：网络图片文字识别

1. icpr_image_slice.py
    训练数据集分为image与txt，运行脚本icpr_image_slice.py 从txt中读取图片标注框的坐标信息，从对应的图片中截取标注部分，并保存，以文本框内容命名，得到清
    洗后的训练数据集train_slice，其中每一个slice均是从原始样本中截取的文本框，并以对应的文本框内容命令；

2. icpr_image_tesseract.py
    使用tesseract tool, 安装中文简体语言包，可对图片进行文字检测：
	import os
	import sys
	os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
	
	import pyocr
	from PIL import Image
	
	tools = pyocr.get_available_tools()[:]
	if(0 == len(tools)):
		print('No usable tools')
		exit(1)
	print(tools[0].image_to_string(Image.open('./demos/自动控温.png'), lang='chi_sim'))
	# 自动控温

3. icpr_tesseract_loss.py
    评估tesseract tool识别精度，对数据要求较高，训练集识别精度40%左右；
    图片进行二值化处理，对识别效果影响不大；

Version：
  OS：Windows7
  python：3.5.2
  numpy: 1.13.1
  tensorflow: 1.4.0
  tesseract: 4.0
  
联系我：hsingmin.lee@yahoo.com

