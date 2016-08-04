#coding:utf-8
# date 写错了
import Get_Date
import theano
import Layer
import numpy as np
import Get_Date
import time
from PIL import Image
images_path = '/home/liuyi/test/1'
image, ans = Get_Date.get_date_i(images_path, 0)
image = [image/255]
ans = np.asarray([ans], np.int32)
#image1, ans1 = Get_Date.get_date_i(images_path, 776)
#image = [image, image1]
# ans = [ans, ans1]
# image_t = image[0]
# print image_t.shape
# image_t = Image.fromarray(image_t)
# image_t.show()
X_shape = (1, 1, 200, 60)
word_len = 20   #单词最长是20
word_len = word_len*2+1
word_kind = 27
# X_shape = (1, 1, images.shape[2], images.shape[3])
model = Layer.SequenceNNs()
model.add(Layer.RCnn_Layer(8, 4, 4, 0.5))
model.add(Layer.RCnn_Layer(16, 4, 4, 0.5))
model.add(Layer.Max_pooling_Layer((2, 2)))
model.add(Layer.RCnn_Layer(16, 4, 4, 0.4))
model.add(Layer.RCnn_Layer(16, 4, 4, 0.4))
#model.add(Layer.Max_pooling_Layer((2, 2)))
#model.add(Layer.RCnn_Layer(32, 4, 4, 0.3))
#model.add(Layer.RCnn_Layer(32, 4, 4, 0.3))
model.add(Layer.Max_pooling_Layer((2, 2)))
model.add(Layer.Flatten_Layer())
#model.add(Layer.Dense_Layer(word_len*word_kind))
#model.add(Layer.Dense_Layer(word_len*word_kind/2))
model.add(Layer.Dense_Layer(word_len))
model.add(Layer.LSTM_Layer(word_kind, 1))
model.add(Layer.LSTM_Layer(word_kind, 1))
model.add(Layer.CTC_Layer(word_len, word_kind))
model.bulid(X_shape)
model.CTC_LOSS()
res = model.predict(image)
loss = model.ctc_loss(image, ans)
#test_grad = model.grad_test(image, ans)
#print image
print ans.shape
print ans
print res
print res.shape
print loss
#print test_grad
#print test_grad.shape
# (index, i_len, j_len) = res.shape
# for i in range(i_len):
#     for j in range(j_len):
#         print res[0][i][j],
#     print "|"
# print loss.shape
# print ans
# (index, i_len, j_len) = loss.shape
# for i in range(i_len):
#     for j in range(j_len):
#         print loss[0][i][j],
#     print "|"
# model.CTC_train()
