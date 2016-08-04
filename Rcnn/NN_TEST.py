#coding:utf-8
'''
Created on 2016年7月8日

@author: Night
'''
import theano
import Layer as Layer
import numpy as np
import Get_Date
import time
import random
import sys

theano.config.floatX = "float32"
save_path = '/home/ly/test/Liuyi/'
images_path = '/home/ly/test/Liuyi/images'
ans_name = 'answer'
#s_sum = Get_Date.get_date_sum(images_path)
s_sum = 10000
start_build = time.clock()
#s_sum = 2
# images, ans = Get_Date.get_date(images_path, ans_name)
# np.save(save_path+'images_images.npy', images)
# np.save(save_path+"images_ans.npy", ans)

images = np.load(save_path+'images_images.npy')
ans = np.load(save_path+"images_ans.npy")
#print "images :{0}".format(s_sum)
#-----------------------------------------------
# images = np.arange(1024).reshape((1, 1, 64, 16)).astype(np.float32)
X_shape = (20, 1, 200, 60)
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
model.CTC_train()
#-----------------------------------------
batch_size = 20
epoch = 2000
blank_index = 0
fill_len = 41
x_sum = int(s_sum * 0.9)
t_sum = s_sum - x_sum
get_random = np.arange(x_sum)
#model.load_weights(save_path+"weight_2.npy")
w_file = "/home/ly/test/Liuyi/w"
end_bulid = time.clock()
print "finish! time = {0}".format(end_bulid - start_build)
for i in range(epoch):
    print "epoch {0}: ".format(i+1)
    random.shuffle(get_random)
    start_epoch = time.clock()
    for j in range(x_sum/batch_size):
        start = time.clock()
        real_index = j*batch_size
        x = [images[m] for m in get_random[real_index: real_index+batch_size]]
        y = [ans[m] for m in get_random[real_index: real_index+batch_size]]
        x = np.array(x, np.float32)
        y = np.array(y, np.int32)
        model.sgd_train(x, y, 1e-2, 0.9)
        loss = np.mean(model.ctc_loss(x, y))
        end = time.clock()
        sys.stdout.write("\r                                                                          ")
        sys.stdout.flush()
        sys.stdout.write("\r{0}/{1} : loss: {2} , time: {3}s".format(real_index + batch_size, x_sum, loss, (end-start)*(-(real_index - x_sum) / batch_size-1)))
        sys.stdout.flush()


    if j*batch_size+batch_size < x_sum:
        start = time.clock()
        x = [images[m] for m in get_random[j*batch_size: x_sum]]
        y = [ans[m] for m in get_random[j*batch_size: x_sum]]
        x = np.array(x, np.float32)
        y = np.array(y, np.int32)
        model.sgd_train(x, y, 1e-2, 0.9)
        loss = np.mean(model.ctc_loss(x, y))
        end = time.clock()
        sys.stdout.write("\r                                                                          ")
        sys.stdout.flush()
        sys.stdout.write("\r{0}/{1} : loss: {2} , time: {3}s".format(x_sum, x_sum, loss, end - start))
        sys.stdout.flush()
    print ""
    loss = 0
    right_sum = 0
    test_index = 9990
    test_x = images[test_index:test_index+1]
    test_y = ans[test_index:test_index+1]
    fp = open(w_file, "a")
    fp.write("epoch{0}:\n".format(i + 1))
    fp.write("\toutput:\n")
    fp.write("\t{0}\n".format(model.predict(test_x)))
    fp.write("\tpredict:\n")
    fp.write("\t{0}\n".format(np.argmax(model.predict(test_x)[0], axis=1)))
    fp.write("\ty:\n")
    fp.write("\t{0}\n".format(test_y))
    fp.write("\tw:\n")
    w_sum = len(model.train_data)
    for s in range(w_sum):
        tmp_data = model.train_data[s].get_value()
        w_len = 5
        if tmp_data.ndim == 1:
            fp.write("\t{0} {1}\n".format(tmp_data[0:w_len], tmp_data.shape))# date 写错了
        if tmp_data.ndim == 2:
            fp.write("\t{0} {1}\n".format(tmp_data[0][0:w_len], tmp_data.shape))  # date 写错了
        if tmp_data.ndim == 3:
            fp.write("\t{0} {1}\n".format(tmp_data[0][0][0:w_len], tmp_data.shape))  # date 写错了
        if tmp_data.ndim == 4:
            fp.write("\t{0} {1}\n".format(tmp_data[0][0][0][0:w_len], tmp_data.shape))  # date 写错了
    fp.close()

    for j in range(t_sum/batch_size):
        real_index = j * batch_size+x_sum
        x = images[real_index: real_index + batch_size]
        y = ans[real_index: real_index + batch_size]
        x = np.array(x, np.float32)
        y = np.array(y, np.int32)
        loss += np.sum(model.ctc_loss(x, y))
        out = model.predict(x)
        right_sum += Get_Date.date_difference(y, out)

    if j * batch_size + batch_size < t_sum:
        x = images[j * batch_size+x_sum: t_sum]
        y = ans[j * batch_size+x_sum: t_sum]
        x = np.array(x, np.float32)
        y = np.array(y, np.int32)
        loss += np.sum(model.ctc_loss(x, y))
        right_sum += Get_Date.date_difference(y, out)

    end_epoch = time.clock()
    print "epoch {0}/{1}: loss: {2} time = {3}s, right_sum:{4}, sum = {5}".format(i+1, epoch, loss/t_sum, end_epoch - start_epoch, right_sum, t_sum)
    model.save_weights(save_path+"weight_3.npy")
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
model.save_weights(save_path+"weight_3.npy")
