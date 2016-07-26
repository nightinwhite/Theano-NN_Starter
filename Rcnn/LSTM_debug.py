# coding:utf-8
'''
Created on 2016年7月8日

@author: Night
'''
import theano
import Layer
import numpy as np


def gen_cosine_amp(amp=100, period=25, x0=0, xn=50000, step=1, k=0.0001):
    cos = np.zeros(((xn - x0) * step)).astype(np.float32)
    for i in range(len(cos)):
        idx = x0 + i * step
        cos[i] = amp * np.cos(idx / (2 * np.pi * period))
        cos[i] = cos[i] * np.exp(-k * idx)
    return cos


xn = 50000
# cos = np.random.randint(1,10,xn).astype(np.float32)
cos = gen_cosine_amp()
expected_output = np.zeros((len(cos))).astype(np.float32)
for i in range(len(cos) - 1):
    expected_output[i] = np.mean(cos[i:i + 2])
cos = np.reshape(cos, (100, -1))
expected_output = np.reshape(expected_output, (100, -1))
slice_pos = 90
X = cos[:slice_pos]
Y = expected_output[:slice_pos]
X_V = cos[slice_pos:]
Y_V = expected_output[slice_pos:]
batch_size = 1
model = Layer.SequenceNNs()
model.add(Layer.LSTM_Layer(hidden_dim=4))
model.bulid((batch_size, X.shape[1]))
model.general_train()
# value = model.test(X_V,Y_V)
# print (value)
# print (value.shape)
loop_num = 100
for i in range(loop_num):
    print("loop: {0}:".format(i))
    slice_sum = X.shape[0] // batch_size
    need_break = False
    for j in range(slice_sum):
        x_s = X[j * batch_size:(j + 1) * batch_size]
        y_s = Y[j * batch_size:(j + 1) * batch_size]
        model.sgd_train(x_s, y_s, 1e-4, 0.9)
        loss = model.get_loss(x_s, y_s)
        loss_v = model.get_loss(X_V, Y_V)
        O_S, Y_S, X_S = model.compare_print(x_s, y_s)
        print ("progress :{0}/{1} loss:{2} loss_v:{3}".format((j + 1) * batch_size, X.shape[0], loss, loss_v))
        print ("O:{0}".format(O_S))
        print ("Y:{0}".format(Y_S))
        print ("X:{0}".format(X_S))
        if (loss_v < 10):
            print (loss_v)
            need_break = True
            break
            # print ("data_d:{0}".format(model.data_d_print(x_s,y_s)))

    if (j + 1) * batch_size < X.shape[0]:
        x_s = X[(j + 1) * batch_size:]
        y_s = Y[(j + 1) * batch_size:]
        model.sgd_train(x_s, y_s, 1e-4, 0.9)
        loss = model.get_loss(X_V, Y_V)
        print ("progress :{0}/{1} loss:{2}".format(X.shape[0], X.shape[0], loss))
    if need_break:
        break
predicted_output = model.predict(X_V)

