#coding:utf-8
'''
Created on 2016年7月18日

@author: HP
'''
import theano
import Layer
import numpy as np
X_shape = (1,50)
word_len = 20#字符的数量
word_kind = 5 #字符的种类
model = Layer.SequenceNNs()
model.add(Layer.CTC_Layer(word_len,word_kind))
model.bulid(X_shape)
model.CTC_LOSS()
x = np.ones(X_shape, dtype = np.float32)
y1 = np.asarray([[2,3,3,3]],dtype = np.int32)
y2 = np.asarray([[2,3,3,3,3,3]],dtype = np.int32)
print (model.ctc_loss(x,y1))
print (model.ctc_loss(x,y2))
