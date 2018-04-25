#coding:utf-8
# ocr_test.py
import ocr.model as om
from glob import glob
import numpy as np
from PIL import Image
import time
import ocr.keys as keys

from keras.utils import plot_model

paths = glob('./test/*.*')

if __name__ =='__main__':
    im = Image.open(paths[1])
    img = np.array(im.convert('RGB'))
    characters = keys.alphabet[:]
    # height = np.shape(img)[0]
    height = 32
    nclass = len(characters)
    model, basemodel = om.get_model(height, nclass)
    print('height = ', height)
    basemodel = om.load_model(height, nclass, basemodel)
    # print(basemodel.summary())
    # print(np.shape(img))
    bimg = im.convert('L')
    scale = bimg.size[1]*1.0/32
    width = int(bimg.size[0]/scale)
    bimg = bimg.resize((width, 32))
    bimg.save('./nbimg.png')
    print(np.shape(bimg))
    print(basemodel.summary())
    result = om.predict(im, basemodel)
    print("---------------------------------------")
    for key in result:
        # print(result[key][1])
        print(result)










