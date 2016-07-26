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

start_t = time.clock()
theano.config.floatX = "float32"
word_len = 20#字符的数量
word_kind = 5 #字符的种类
Y = T.imatrix()#必须是INT类型的

def test(word_len,Y,output,output_shape):
    T_ctc = word_len#字符的数量
    L = Y.shape[1] * 2 + 1#答案的字符长度
    output = output
    y = Y[0]
    def extend_y(i, y):
        return T.switch(T.eq(i % 2, 0), 0, y[(i - 1) // 2])
    y, _ = theano.scan(extend_y, sequences=[T.arange(L)], non_sequences=[y])
    return y 

def CTC_LOSS1(word_len,Y,output,output_shape):
        T_ctc = word_len#字符的数量
        L = Y.shape[1] * 2 + 1#答案的字符长度
        output = output
        def each_loss(index, T_ctc, L,output,Y):
            o = output[index]
            y = Y[index]
            blank_num = 0
            def extend_y(i, y):
                return T.switch(T.eq(i % 2, 0), blank_num, y[(i - 1) // 2])
            y, _ = theano.scan(extend_y, sequences=[T.arange(L)], non_sequences=[y])
            #y扩大为2*y.len+1由blank_num填充
            temp_vector = T.zeros(output_shape[1]*2+1)
            alpha0 = T.concatenate([[o[0][y[0]]], [o[0][y[1]]], T.zeros_like(temp_vector[:L-2])],axis = 0)
            #return alpha0
            #！！！
            #alpha0是第一层的答案

            def to_T(t, alpha_pre, o, y, T_ctc, L):#对一层进行构造
                alpha_e = 1 + 2*t
                alpha_b = L - 2*T_ctc+2*t
                def set_alpha_value(i,alpha_t,alpha_pre,t,o,y):#对层的单一节点赋值
                    iff = T.cast(0,dtype = "float32")
                    ift = (alpha_pre[i] + T.gt(i, 0) * alpha_pre[i - 1] + (T.gt(i, 1) * T.eq(i % 2, 1)) * alpha_pre[i - 2]) * o[t][y[i]]
                    ans = theano.ifelse.ifelse(T.eq(alpha_t[i],1),ift,iff)
                    return ans

                temp_vector = T.zeros(output_shape[1]*2+1)
                alpha_v = T.ones_like(temp_vector[:(T.switch(T.gt(alpha_e, L - 1), L - 1, alpha_e) - T.switch(T.gt(alpha_b, 0), alpha_b, 0))+1])
                alpha_t = theano.ifelse.ifelse(T.gt(alpha_b, 0), T.concatenate([T.zeros_like(temp_vector[:alpha_b]), alpha_v]), alpha_v)
                alpha_t = theano.ifelse.ifelse(T.ge(alpha_e, L - 1), alpha_t, T.concatenate([alpha_t,T.zeros_like(temp_vector[:L-1-alpha_e])]))
                alpha_t = theano.scan(set_alpha_value,
                                      sequences=[T.arange(alpha_t.shape[0])],
                                      non_sequences=[alpha_t,alpha_pre,t,o,y])
                return alpha_t
#             alphas, _ = theano.scan(to_T, sequences=[T.arange(1, T_ctc)],
#                                    outputs_info=[alpha0],
#                                    non_sequences=[o, y, T_ctc, L])
            alphas, _ = theano.scan(to_T, sequences=[T.arange(1, T_ctc)],
                                    outputs_info = [alpha0],
                                   non_sequences=[o, y, T_ctc, L])
            loss = alphas[-1][-1] + alphas[-1][-2]
            loss = T.switch(T.le(loss, 1e-45), 1e-45, loss)
            loss = -T.log(loss)
            return loss

        CTC_LOSSs, _ = theano.scan(each_loss,
                                  sequences=[T.arange(output_shape[0])],
                                  non_sequences=[T_ctc, L,output,Y])
        return CTC_LOSSs
def CTC_LOSS(outpts, inpts):

    def each_loss(outpt, inpt):
        #y 是填充了blank之后的ans
        blank = 0
        y_nblank = T.neq(inpt, blank)
        n = T.dot(y_nblank, y_nblank)#真实的字符长度
        N = 2*n+1#填充后的字符长度，去除尾部多余的填充
        labels = inpt[:N]
        labels2 = T.concatenate((labels, [blank, blank]))
        sec_diag = T.neq(labels2[:-2], labels2[2:]) * T.eq(labels2[1:-1], blank)
        recurrence_relation = \
            T.eye(N) + \
            T.eye(N, k=1) + \
            T.eye(N, k=2) * sec_diag.dimshuffle((0, 'x'))

        pred_y = outpt[:, labels]

        fwd_pbblts, _ = theano.scan(
            lambda curr, accum: curr * T.dot(accum, recurrence_relation),
            sequences=[pred_y],
            outputs_info=[T.eye(N)[0]]
        )
        liklihood = fwd_pbblts[-1, -1]+fwd_pbblts[-1, -2]
        loss = -T.log(liklihood)
        return loss
        #return N
    ctc_losss, _ = theano.scan(each_loss,
                               sequences=[outpts, inpts],
                               )
    return ctc_losss


X = T.matrix()
input_shape = (2,50)
hidden_dim = 5
l = Layer.CTC_Layer(Word_len = word_len,Word_kind = word_kind)
output_shape = (input_shape[0],word_len,word_kind)       
l.bulid(input_shape)
l.set_input(X)
O = l.get_output()
x = np.arange(input_shape[0]*input_shape[1]).reshape(input_shape).astype(np.float32)
f = theano.function([X], O)
softmax_O = f(x)
softmax_O = np.full(softmax_O.shape, 0.5, dtype = np.float32)
softmax_O += np.eye(word_len, word_kind)/2
softmax_O[0, :, 0] = 1
print (softmax_O.shape)
y = np.asarray([[0, 4, 0, 1, 0, 2, 0, 0, 0], [0, 1, 0, 2, 0, 3, 0, 4, 0]], dtype = np.int32)
#CTC_O = CTC_LOSS1(word_len, Y, O, output_shape)
CTC_O = CTC_LOSS(O, Y)
t_1 = time.clock()
f = theano.function([O, Y], CTC_O)
print (f(softmax_O, y))
end_t = time.clock()
print end_t - start_t
print end_t - t_1
