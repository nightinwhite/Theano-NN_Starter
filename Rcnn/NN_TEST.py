#coding:utf-8
'''
Created on 2016年7月8日

@author: Night
'''
import theano
import Layer
import numpy as np
import Get_Date
import time
import random

theano.config.floatX = "float32"
save_path = '/home/liuyi/test/'
images_path = '/home/liuyi/test/captcha'
s_sum = Get_Date.get_date_sum(images_path)
start_build = time.clock()
#s_sum = 2
images, ans = Get_Date.get_date(images_path)
np.save(save_path+'captcha_images.npy', images)
np.save(save_path+"captcha_ans.npy", ans)

# images = np.load(save_path+'captcha_images.npy')
# ans = np.load(save_path+"captcha_ans.npy")

#print "images :{0}".format(s_sum)
#-----------------------------------------------
# images = np.arange(1024).reshape((1, 1, 64, 16)).astype(np.float32)
X_shape = (1, 1, 200, 60)
word_len = 50   #单词最长是46
word_kind = 59
# X_shape = (1, 1, images.shape[2], images.shape[3])
model = Layer.SequenceNNs()
model.add(Layer.RCnn_Layer(32, 4, 4))
# model.add(Layer.RCnn_Layer(32, 4, 4))
# model.add(Layer.Max_pooling_Layer((2, 2)))
# model.add(Layer.RCnn_Layer(32, 4, 4))
# model.add(Layer.RCnn_Layer(32, 4, 4))
# model.add(Layer.Max_pooling_Layer((2, 2)))
# model.add(Layer.RCnn_Layer(32, 4, 4))
# model.add(Layer.RCnn_Layer(32, 4, 4))
model.add(Layer.Max_pooling_Layer((2, 2)))
model.add(Layer.Flatten_Layer())
model.add(Layer.Dense_Layer(2048))
model.add(Layer.LSTM_Layer())
# model.add(Layer.LSTM_Layer())
model.add(Layer.CTC_Layer(word_len, word_kind))
model.bulid(X_shape)
model.CTC_train()

#-----------------------------------------
batch_size = 1
epoch = 10
end_bulid = time.clock()
blank_index = 0
fill_len = 93
print "finish! time = {0}".format(end_bulid - start_build)
x_sum = int(s_sum * 0.9)
t_sum = s_sum - x_sum
get_random = np.arange(s_sum)
#model.load_weights("/home/liuyi/test/weight_1.npy")
for i in range(epoch):
    print "epoch {0}: ".format(i)
    random.shuffle(get_random)
    start_epoch = time.clock()
    for j in range(x_sum/batch_size):
        start = time.clock()
        real_index = j*batch_size
        x = [images[m] for m in get_random[real_index: real_index+batch_size]]
        y = [ans[m] for m in get_random[real_index: real_index+batch_size]]
        x = np.array(x, np.float32)
        y = np.array(y, np.int32)
        model.sgd_train(x, y, 1e-4, 0.9)
        loss = np.mean(model.ctc_loss(x, y))
        end = time.clock()
        print "{0}/{1} : loss: {2} , time: {3}s".format(real_index + batch_size, x_sum, loss, end-start)
    if j*batch_size+batch_size < x_sum:
        start = time.clock()
        x = [images[m] for m in get_random[j*batch_size: x_sum]]
        y = [ans[m] for m in get_random[j*batch_size: x_sum]]
        x = np.array(x, np.float32)
        y = np.array(y, np.int32)
        model.sgd_train(x, y, 1e-4, 0.9)
        loss = np.mean(model.ctc_loss(x, y))
        end = time.clock()
        print "{0}/{1} : loss: {2} , time: {3}s".format(x_sum, x_sum, loss, end - start)

    loss = 0

    for j in range(t_sum/batch_size):
        real_index = j * batch_size+x_sum
        x = [images[m] for m in get_random[real_index: real_index + batch_size]]
        y = [ans[m] for m in get_random[real_index: real_index + batch_size]]
        x = np.array(x, np.float32)
        y = np.array(y, np.int32)
        loss += np.sum(model.ctc_loss(x, y))

    if j * batch_size + batch_size < t_sum:
        x = [images[m] for m in get_random[j * batch_size+x_sum: t_sum]]
        y = [ans[m] for m in get_random[j * batch_size+x_sum: t_sum]]
        x = np.array(x, np.float32)
        y = np.array(y, np.int32)
        loss += np.sum(model.ctc_loss(x, y))
    end_epoch = time.clock()
    print "epoch {0}/{1}: loss: {2} time = {3}s,".format(i, epoch, loss/t_sum, end_epoch - start_epoch)

# for i in range(epoch):
#     print "epoch {0}: ".format(i)
#     start_epoch = time.clock()
#     for j in range(x_sum/batch_size):
#         start = time.clock()
#         real_index = j*batch_size
#         x = []
#         y = []
#         for m in range(real_index, real_index+batch_size):
#             image, ans = Get_Date.get_date_i(images_path, m)
#             x.append(image)
#             ans = Get_Date.fill_blank(ans, blank_index, fill_len)
#             y.append(ans)
#
#         x = np.array(x, np.float32)
#         y = np.array(y, np.int32)
#         model.sgd_train(x, y, 1e-4, 0.9)
#         loss = model.ctc_loss(x, y)
#         end = time.clock()
#         print "{0}/{1} : loss: {2} , time: {3}s".format(real_index + batch_size, x_sum, loss, end-start)
#     if j*batch_size+batch_size < x_sum:
#         start = time.clock()
#         for m in range(j*batch_size, x_sum):
#             image, ans = Get_Date.get_date_i(images_path, m)
#             x.append(image)
#             ans = Get_Date.fill_blank(ans, blank_index, fill_len)
#             y.append(ans)
#         x = np.array(x, np.float32)
#         y = np.array(y, np.int32)
#         model.sgd_train(x, y, 1e-4, 0.9)
#         loss = model.ctc_loss(x, y)
#         end = time.clock()
#         print "{0}/{1} : loss: {2} , time: {3}s".format(x_sum, x_sum, loss, end - start)
#
#     loss = 0
#
#     for j in range(t_sum/batch_size):
#         real_index = j * batch_size+x_sum
#         x = []
#         y = []
#         for m in range(real_index, real_index + batch_size):
#             image, ans = Get_Date.get_date_i(images_path, m)
#             x.append(image)
#             ans = Get_Date.fill_blank(ans, blank_index, fill_len)
#             y.append(ans)
#         x = np.array(x, np.float32)
#         y = np.array(y, np.int32)
#         loss += model.ctc_loss(x, y)
#
#     if j * batch_size + batch_size < t_sum:
#         for m in range(j * batch_size+x_sum, t_sum):
#             image, ans = Get_Date.get_date_i(images_path, m)
#             x.append(image)
#             ans = Get_Date.fill_blank(ans, blank_index, fill_len)
#             y.append(ans)
#         x = np.array(x, np.float32)
#         y = np.array(y, np.int32)
#         loss += model.ctc_loss(x, y)
#     end_epoch = time.clock()
#     print "epoch {0}/{1}: loss: {2} time = {3}s,".format(i, epoch, loss/t_sum, end_epoch - start_epoch)
model.save_weights("/home/liuyi/test/weight_1.npy")
