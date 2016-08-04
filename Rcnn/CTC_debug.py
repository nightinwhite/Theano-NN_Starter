#coding:utf-8
'''
Created on 2016年7月18日

@author: HP
'''
from theano import tensor as T
import theano 
import numpy as np
import Layer
import time
import Get_Date

X_shape = (1, 1, 200, 60)
word_len = 46   #单词最长是46
word_kind = 59
save_path = "/home/liuyi/test/"
images_path = '/home/liuyi/test/captcha'
images, ans = Get_Date.get_date_i(images_path, 20)
images = np.asarray([images], np.float32)
ans = np.asarray([ans], np.int32)
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
model.CTC_LOSS()
model.load_weights(save_path+"weight_2.npy")
softmax_O = model.predict(images)

(index, i_len, j_len) = softmax_O.shape
for i in range(i_len):
    for j in range(j_len):
        print "%5.2f" % softmax_O[0][i][j],
    print

#softmax_O = np.full(softmax_O.shape, 0, dtype = np.float32)
# softmax_O += np.eye(word_len, word_kind)
# softmax_O[0, :, 0] = 1
y = Get_Date.fill_blank([1, 2, 3, 4, 5], 0, 49)
y = [y]
#(i_len, j_len) = softmax_O.shape
# for i in range(i_len):
#     for j in range(j_len):
#         print softmax_O[i][j],
#     print
#a = model.ctc_loss(images, ans)
a = model.test_cte(softmax_O, ans)
print a

# (index, i_len, j_len) = a.shape
# for i in range(i_len):
#     for j in range(j_len):
#         print a[0][i][j],
#     print

#print b
#print a.shape
#print b.shape