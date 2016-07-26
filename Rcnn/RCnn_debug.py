#coding:utf-8
'''
Created on 2016年7月19日

@author: HP
'''
from theano import tensor as T
from PIL import Image
import theano 
import numpy as np
import Layer

theano.config.floatX = "float32"
X = T.tensor4()
input_shape = (1,1,3,6)
l = Layer.RCnn_Layer(nb_filter=3,nb_row=2,nb_col=2)
l.bulid(input_shape)
l.set_input(X)
O = l.get_output()
f = theano.function([X],O)

#image = Image.open("/home/liuyi/test/1.jpg").convert("L")
image = np.arange(input_shape[0]*input_shape[1]*input_shape[2]*input_shape[3]).reshape(input_shape)
image = np.asarray(image, np.float32)
#image = np.asarray([[image]], np.float32)
print image.shape
image_o = f(image)
# print test_ans1
# print test_ans2
# print test_ans3
print image_o
print image_o.shape
# image_o = image_o.astype(np.uint8)
# #image_o = image_o.transpose((0, 2, 3, 1))
# image_f = Image.fromarray(image_o[0][0])
# image_f.save("/home/liuyi/test/2_1.jpg")
# image_f = Image.fromarray(image_o[0][1])
# image_f.save("/home/liuyi/test/2_2.jpg")
# image_f = Image.fromarray(image_o[0][2])
# image_f.save("/home/liuyi/test/2_3.jpg")
#print (image.shape)
