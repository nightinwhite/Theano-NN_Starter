#coding:utf-8
import Get_Date
import theano
import Layer
import numpy as np
import Get_Date
import time
from PIL import Image
images_path = '/home/liuyi/test/captcha'
image, ans = Get_Date.get_date_i(images_path, 777)
#image = [image]
image1, ans1 = Get_Date.get_date_i(images_path, 776)
image = [image, image1]
# ans = [ans, ans1]
# image_t = image[0]
# print image_t.shape
# image_t = Image.fromarray(image_t)
# image_t.show()
X_shape = (1, 1, 200, 60)
word_len = 50   #单词最长是46
word_kind = 59
# X_shape = (1, 1, images.shape[2], images.shape[3])
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
model.add(Layer.Dense_Layer(word_len))
model.add(Layer.LSTM_Layer(10))
# model.add(Layer.LSTM_Layer())
model.add(Layer.CTC_Layer(word_len, word_kind))
model.bulid(X_shape)
res = model.predict(image)
print image
print res
print res.shape
# model.CTC_train()
