#coding:utf-8
'''
Created on 2016年7月26日

@author: Night
'''
import theano
import Layer as Layer
import numpy as np
import Get_Date
import time
import random
import sys

X_shape = (200, 1, 200, 60)
word_len = 46   #单词最长是46
word_kind = 59
save_path = "/home/liuyi/test/"
images_path = '/home/liuyi/test/captcha'
model = Layer.SequenceNNs()
model.add(Layer.RCnn_Layer(8, 4, 4))
model.add(Layer.RCnn_Layer(16, 4, 4))
model.add(Layer.Max_pooling_Layer((2, 2)))
model.add(Layer.RCnn_Layer(16, 4, 4))
model.add(Layer.RCnn_Layer(16, 4, 4))
model.add(Layer.Max_pooling_Layer((2, 2)))
model.add(Layer.RCnn_Layer(32, 4, 4))
model.add(Layer.RCnn_Layer(32, 4, 4))
model.add(Layer.Max_pooling_Layer((2, 2)))
model.add(Layer.Flatten_Layer())
model.add(Layer.Dense_Layer(word_len*2+1))
model.add(Layer.LSTM_Layer(10))
model.add(Layer.LSTM_Layer(10))
model.add(Layer.CTC_Layer(word_len, word_kind))
model.bulid(X_shape)
#model.CTC_train()
model.load_weights(save_path+"weight_2.npy")
test_len = 100
images = []
ans_s = []
for i in range(test_len):
    image_i, ans_i = Get_Date.get_date_i(images_path, i)
    images.append(image_i)
    ans_s.append(ans_i)
images = np.asarray(images, np.float32)
ans_s = np.asarray(ans_s, np.int32)
y = model.predict(images)
print y
print images.shape
print y.shape
print Get_Date.date_difference(ans_s, y)
