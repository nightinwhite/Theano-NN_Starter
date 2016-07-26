#coding:utf-8
'''
Created on 2016年6月6日

@author: Night
'''
import theano
import numpy as np
from abc import abstractmethod
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import sandbox
class Layer(object):
    def __init__(self):
        pass

    @abstractmethod
    def train(self,real_output):
        pass

    @abstractmethod
    def predict(self,input):
        pass#这里可以保存最新的输出，减少计算量

    @abstractmethod
    def bulid(self,input_shape):
        pass#这里是一些待模型建立好才能确定的值的初始化

class RCnn_Layer(Layer):
    ori_image = None
    def __init__(self, nb_filter, nb_row, nb_col,dropout_rate = 0,border_mode='valid',Rnn_Way ="this"):
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.dropout_rate = dropout_rate
        self.border_mode = border_mode
        #这里默认为valid
        assert Rnn_Way in {'ori','bef','this'},'Rnn_Way must be in {ori,bef,this}'
        self.activition = T.nnet.relu

    def  bulid(self,input_shape):
        self.input_shape = input_shape # (batch size, input channels, input rows, input columns).

        mu = 0
        sigma = 1
        #scale_b = 3
        shape = (self.nb_filter, self.input_shape[1],self.nb_row,self.nb_col)#(output channels, input channels, filter rows , filter columns)
        self.Cnn_W = theano.shared(np.random.normal(mu, sigma, size = shape).astype(np.float32), borrow=True)#(output channels, input channels, filter rows , filter columns).
        self.Cnn_B = theano.shared(np.zeros(self.nb_filter).astype(np.float32), borrow=True)#(output channels,).
        #self.Rnn_W_s = theano.shared( np.random.uniform(low = 0, high = 1 , size=(self.nb_filter)).astype(np.float32))#先设置为一个值,之前的状态,对应每个feature map
        #self.Rnn_W_s = theano.shared(np.full((self.nb_filter), 1, np.float32))
        #self.Rnn_W_x = theano.shared( np.random.uniform(low = - scale, high = scale , size=(self.nb_filter)).astype(np.float32))#先设置为一个值，当前输入，对应每个feature map
        #self.Rnn_W_b = theano.shared( np.random.uniform(low = 0, high = scale_b , size=(self.nb_filter)).astype(np.float32))#先设置为一个值，bias，对应每个feature map

    def set_input(self,input):
        self.input = input

    def get_output(self):
        if self.dropout_rate!=0:
            seed = np.random.randint(10e6)
            rng = RandomStreams(seed=seed)
            retain_prob = 1. - self.dropout_rate
            self.input *= rng.binomial(self.input.shape, p=retain_prob, dtype=self.input.dtype)
            self.input /= retain_prob
        conv_out = conv2d(self.input,self.Cnn_W) #(batch size, output channels, output rows, output columns)
        conv_out = conv_out + self.Cnn_B.dimshuffle('x', 0, 'x', 'x')
        # out_put_shape = self.get_output_shape()
        # r_matrix_s = np.eye(out_put_shape[3], out_put_shape[3], 0)
        # r_matrix_x = np.eye(out_put_shape[3], out_put_shape[3], -1)
        # test = [[r_matrix_s for i in range(self.input_shape[1])] for j in range(self.input_shape[0])]
        # print test
        # r_matrix_s = theano.shared(np.array(r_matrix_s).astype(np.float32))
        #
        # r_matrix_x = theano.shared(np.array(r_matrix_x).astype(np.float32))
        #
        # r_matrix = r_matrix_s*self.Rnn_W_s.dimshuffle(0, 'x', 'x') + \
        #             r_matrix_x*(1-self.Rnn_W_s).dimshuffle(0, 'x', 'x')
        # conv_out = conv_out.dimshuffle(1, 0, 2, 3)
        # def step (con, r_m, r_b):
        #     return T.dot(con, r_m) + r_b
        # conv_out, _ = theano.scan(step, sequences=[conv_out, r_matrix, self.Rnn_W_b])
        # conv_out = conv_out.dimshuffle(1, 0, 2, 3)
        # R_conv_out = T.concatenate([T.zeros_like(conv_out[:, :, :, :1]), conv_out], axis = 3)
        # R_conv_out = R_conv_out[:, :, :,:conv_out.shape[3]]
        # RNN_Ws = self.Rnn_W_s.dimshuffle('x', 0, 'x', 'x')
        # RNN_b = self.Rnn_W_b
        # R_conv_out = R_conv_out *RNN_Ws + conv_out * (1-RNN_Ws) + RNN_b
        # conv_out = conv_out.dimshuffle(1,0,2,3)
        #
        # def Rnn_add(channel,RNN_b,RNN_Ws,RNN_Wx):
        #     RNN_channel = T.concatenate([T.zeros_like(channel[:, :, :1]),channel],axis = 2)
        #     RNN_channel = RNN_channel[:,:,:channel.shape[2]]
        #     res = RNN_channel*RNN_Ws + channel*RNN_Wx + RNN_b
        #     return res
        #self.Rnn_W_s = T.abs_(self.Rnn_W_s)
        # R_conv_out,_ = theano.scan(Rnn_add,sequences= [conv_out,self.Rnn_W_b,self.Rnn_W_s,1 - self.Rnn_W_s])
        # R_conv_out = R_conv_out.dimshuffle(1,0,2,3)
        #output = self.activition(R_conv_out)
        #return self.input
        return self.activition(conv_out)
        #return output

    def get_train_data(self):
        return [self.Cnn_W, self.Cnn_B]
        #return [self.Cnn_W, self.Cnn_B, self.Rnn_W_s, self.Rnn_W_b]
    def conv_output_length(self,input_length, filter_size, border_mode = 'valid', stride = 1):
        if input_length is None:
            return None
        assert border_mode in {'same', 'valid'}
        if border_mode == 'same':
            output_length = input_length
        elif border_mode == 'valid':
            output_length = input_length - filter_size + 1
        return (output_length + stride - 1) // stride

    def get_output_shape(self):
        rows = self.input_shape[2]
        cols = self.input_shape[3]
        rows = self.conv_output_length(rows, self.nb_row,
                                  self.border_mode, 1)
        cols = self.conv_output_length(cols, self.nb_col,
                                  self.border_mode, 1)
        return (self.input_shape[0], self.nb_filter, rows, cols)

class Max_pooling_Layer(Layer):
    def __init__(self,pool_size):
        self.pool_size = pool_size

    def bulid(self,input_shape):
        self.input_shape = input_shape

    def set_input(self,input):
        self.input = input

    def get_output(self):
        return pool.pool_2d(self.input, self.pool_size, ignore_border=True)

    def get_output_shape(self):
        rows = self.input_shape[2]
        cols = self.input_shape[3]
        rows = rows//self.pool_size[0]
        cols = cols//self.pool_size[1]
        return (self.input_shape[0], self.input_shape[1], rows, cols)

    def get_train_data(self):
        return None

class Flatten_Layer(Layer):
    def __init__(self):
        pass

    def bulid(self,input_shape):
        self.input_shape = input_shape

    def set_input(self,input):
        self.input = input

    def get_output(self):
        return T.reshape(self.input, (self.input.shape[0], T.prod(self.input.shape) // self.input.shape[0]))

    def get_output_shape(self):
        input_shape = self.input_shape
        if not all(input_shape[1:]):
            raise Exception('The shape of the input to "Flatten" '
                            'is not fully defined '
                            '(got ' + str(input_shape[1:]) + '. '
                            'Make sure to pass a complete "input_shape" '
                            'or "batch_input_shape" argument to the first '
                            'layer in your model.')
        return (input_shape[0], np.prod(input_shape[1:]))

    def get_train_data(self):
        return None

class Dense_Layer(Layer):
    def __init__(self,output_dim):
        self.output_dim = output_dim
        self.activation = T.nnet.relu
        pass

    def bulid(self,input_shape):
        self.input_shape = input_shape
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        mu = (self.output_dim+0.0) / input_dim
        sigma = 0.01
        self.w = theano.shared(np.random.uniform(mu, sigma, size=(input_dim,)).astype(np.float32), borrow=True)
        self.b = theano.shared(np.zeros(self.output_dim).astype(np.float32), borrow=True)

    def set_input(self,input):
        self.input = input

    def get_output(self):
        w_mask = T.zeros([self.input_shape[1], self.output_dim])
        output = T.dot(self.input, w_mask+self.w.dimshuffle(0, 'x'))
        output += self.b
        return self.activation(output)

    def get_output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_train_data(self):
        return [self.w, self.b]

class LSTM_Layer(Layer):
    def __init__(self,hidden_dim = 128, bptt_truncate = -1):
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        #self.input_shape = None
        #self.input = None
        pass  
    def bulid(self,input_shape):
        self.input_shape = input_shape
        self.inpt_ndim = self.input_shape[2] if len(self.input_shape) == 3 else 1
        #print self.inpt_ndim
        mu = 1. / self.hidden_dim
        sigma = 0.1
        #self.E = theano.shared(np.random.normal(mu, sigma, self.hidden_dim).astype(np.float32), borrow=True)#-np.sqrt(1./self.hidden_dim),np.sqrt(1./self.hidden_dim)
        self.U = theano.shared(np.random.normal(mu, sigma, (4, self.inpt_ndim, self.hidden_dim, )).astype(np.float32), borrow=True)
        self.W = theano.shared(np.random.normal(mu, sigma, (4, self.hidden_dim, self.hidden_dim)).astype(np.float32), borrow=True)
        #self.V = theano.shared(np.random.normal(mu, sigma, (self.hidden_dim, self.inpt_ndim)).astype(np.float32), borrow=True)
        self.b = theano.shared(np.zeros((4, self.hidden_dim), dtype=np.float32), borrow=True)
        #self.c = theano.shared(np.zeros(1, dtype=np.float32), borrow=True)
        pass
    
    def step(self, t, s_p, c_p, X):
        #x_t = X[:,t]
        #X = T.matrix()
        if len(self.input_shape) == 3:
            x_t = X[:, t]
        else:
            x_t = X[:, t:t+1]
        #x_t = X[:,t+self.input_shape[1]-self.hidden_dim+1:t+self.input_shape[1]+1]
        #x_t = x_t*self.E
        #test = T.dot(x_t, self.U)
        res_s = T.dot(x_t, self.U) + T.dot(s_p, self.W) + self.b#[index,channel,hidden_dim]
        i = T.nnet.hard_sigmoid(res_s[:, 0, :])# (index,hidden_dim)
        f = T.nnet.hard_sigmoid(res_s[:, 1, :])#(index,hidden_dim)
        o = T.nnet.hard_sigmoid(res_s[:, 2, :])#(index,hidden_dim)
        g = T.tanh(res_s[:, 3, :])#(index,hidden_dim)
        # i = T.nnet.hard_sigmoid(T.dot(x_t, self.U[0])+T.dot(s_p,self.W[0])+self.b[0])#(index,hidden_dim)
        # f = T.nnet.hard_sigmoid(T.dot(x_t, self.U[1])+T.dot(s_p,self.W[1])+self.b[1])#(index,hidden_dim)
        # o = T.nnet.hard_sigmoid(T.dot(x_t, self.U[2])+T.dot(s_p,self.W[2])+self.b[2])#(index,hidden_dim)
        # g = T.tanh(T.dot(x_t, self.U[3])+T.dot(s_p,self.W[3])+self.b[3])#(index,hidden_dim)
        c_t = c_p*f + g*i#(index,hidden_dim)
        s_t = T.tanh(c_t)*o#(index,hidden_dim)
        # o_t = T.dot(s_t, self.V)#(index,1)
        # o_t = o_t+self.c[0]
        o_t = s_t
        #return o_t
        # o_t = T.cast(o_t,"float32")
        # s_t = T.cast(s_t,"float32")
        # c_t = T.cast(c_t, "float32")
        return [o_t, s_t, c_t]
        #return [o_t,s_t,c_t]

    def test(self,t,X):
        X_T = T.concatenate([X,X],axis=1) 
        res = X_T[:,t:t+self.hidden_dim]                   
        return res
    
    def set_input(self,input):
        self.input = input
        
    def get_output(self):
        #X = T.concatenate([self.input, self.input], axis = 1)#!!!!这里修改了LSTM的原理
        X = self.input
        [o,s,c],upd = theano.scan(self.step,sequences=[T.arange(self.input_shape[1])],
                                  outputs_info = [None,dict(initial = T.zeros((self.input.shape[0], self.hidden_dim), "float32")), dict(initial = T.zeros((self.input.shape[0], self.hidden_dim), "float32"))]
                                  , non_sequences = [X])
        o = o.dimshuffle(1, 0, 2)
        return o
        
    def get_output_shape(self):
        return (self.input_shape[0], self.input_shape[1], self.hidden_dim)
    def get_train_data(self):
        return [self.U, self.W, self.b]
        #return [self.E, self.U, self.W, self.V, self.b, self.c]
    
class CTC_Layer(Layer):#需要处理成三维的（batchsize,Vector）->(batchsize,Word_len,Word_kind)
    def __init__(self,Word_len,Word_kind):
        self.word_len = Word_len
        self.word_kind = Word_kind
        
    def bulid(self, input_shape):
        self.input_shape = input_shape
        self.vector_len = input_shape[1]
        self.fstep =self.vector_len/self.word_len
        self.data_len = int(self.fstep)+1
        self.need_len = int((self.word_len-1)*self.fstep+0.5)+self.data_len-1
        self.add_len = self.need_len -self.data_len
        mu = 0
        sigma = 0.01
        if len(self.input_shape) == 2:
            self.S_W = theano.shared(np.random.normal(mu, sigma, (self.word_kind, self.data_len)).astype(np.float32), borrow=True)
        else:
            self.S_W = theano.shared(
                np.random.normal(mu, sigma, (self.input_shape[2], self.word_kind)).astype(
                    np.float32), borrow=True)
        self.S_C = theano.shared(np.random.normal(mu, sigma, (self.word_kind)).astype(np.float32), borrow=True)
    
    def set_input(self,input):
        self.input = input
        
    def CTC_reshape(self,Vector):
        con_Vector = Vector[self.vector_len-self.add_len:]
        con_Vector = T.zeros_like(con_Vector)
        X_Vector = T.concatenate([con_Vector,Vector])
        def mini_reshape(index,X):
            real_index = T.cast((index*self.fstep+0.5),'int32')
            res = X[real_index+self.add_len+1-self.data_len:real_index+self.add_len+1]
            res_softmax = T.nnet.softmax(self.S_W.dot(res)+self.S_C)
            res_softmax = res_softmax[0]
            return res_softmax
        res,upd = theano.scan(mini_reshape,sequences=[T.arange(self.word_len)],non_sequences = [X_Vector])
        return res
    
    def get_output(self):
        if len(self.input_shape) == 2:
            output, _ = theano.scan(self.CTC_reshape, sequences=[self.input])#self.input->(batch_size, T_len)
        else:
            output, _ = theano.scan(lambda x, w, c: T.nnet.softmax(T.dot(x, w)+c), sequences=[self.input],
                                 non_sequences=[self.S_W, self.S_C])
            #output = T.nnet.softmax(T.dot(self.input, self.S_W)+self.S_C)#self.input->(batch_size, T_len, hidden_dim)
        return output
    def get_output_shape(self):
        return (self.input_shape[0],self.word_len, self.word_kind)
       
    def get_train_data(self):
        return [self.S_W,self.S_C]
    
    
class SequenceNNs():
    def __init__(self):
        self.model = []
        theano.config.floatX = "float32"
        self.X = T.tensor4()
        self.Y = T.imatrix()#Y(batchsize,vector)
        # self.X = T.tensor4()
        # self.Y = T.imatrix()
        # self.X = T.matrix()
        # self.Y = T.imatrix()
        self.train_data = []
        
    def add(self,newLayer):
        self.model.append(newLayer)
        
    def bulid(self,input_shape):
        temp_input = input_shape
        for NN in self.model:
            NN.bulid(temp_input)
            temp_input = NN.get_output_shape()

        self.output_shape = temp_input
        self.output = self.get_output()
        self.predict = theano.function([self.X], self.output)
        
    def get_output(self):
        temp_input = self.X
        for NN in self.model:
            NN.set_input(temp_input)
            temp_input = NN.get_output()
            #print type(temp_input)
            temp_train_data = NN.get_train_data()
            if temp_train_data is not None:
                for data in temp_train_data:
                    self.train_data.append(data)
        return temp_input

    def save_weights(self, file_path):
        save_arr = []
        for W in self.train_data:
            save_arr.append(W.get_value())
        np.save(file_path, save_arr)

    def load_weights(self, file_path):
        load_arr = np.load(file_path)
        for i in range(len(self.train_data)):
            self.train_data[i].set_value(load_arr[i])


    def CTC_B(self,A):
        blank_num = 0
        i = len(A) -1 
        j = i
        while i != 0 :
            j = i-1
            if A[i]!=blank_num and A[j] == A[i]:
                del A[i]
            elif A[i] == blank_num:
                del A[i]
            i-=1
        if A[0] == blank_num :
            del A[0]
        return A
    
    # def CTC_LOSS(self):
    #     T_ctc = self.output_shape[1]#字符的数量
    #     L = self.Y.shape[1]*2+1#答案的字符长度
    #
    #     def each_loss(index,T_ctc,L):
    #         o = self.output[index]
    #         y = self.Y[index]
    #         blank_num = 0
    #         def extend_y(i,y):
    #             return T.switch(T.eq(i%2, 0), blank_num, y[(i-1)//2])
    #         y,_ = theano.scan(extend_y,sequences=[T.arange(L)],non_sequences = [y])
    #         #y扩大为2*y.len+1由blank_num填充
    #         temp_vector = T.zeros(self.output_shape[1]*2+1)
    #         alpha0 = T.concatenate([[o[0][y[0]]], [o[0][y[1]]], T.zeros_like(temp_vector[:L-2])],axis = 0)
    #         #alpha0是第一层的答案
    #         def to_T(t,alpha_pre,o,y,T_ctc,L):#对一层进行构造
    #             alpha_e = 1 + 2*t
    #             alpha_b = L - 2*T_ctc+2*t
    #             def set_alpha_value(i,alpha_t,alpha_pre,t,o,y):#对层的单一节点赋值
    #                 iff = T.cast(0,dtype = "float32")
    #                 ift = (alpha_pre[i] + T.gt(i, 0) * alpha_pre[i - 1] + (T.gt(i, 1) * T.eq(i % 2, 1)) * alpha_pre[i - 2]) * o[t][y[i]]
    #                 ans = theano.ifelse.ifelse(T.eq(alpha_t[i],1),ift,iff)
    #                 return ans
    #
    #             temp_vector = T.zeros(self.output_shape[1]*2+1)
    #             alpha_v = T.ones_like(temp_vector[:(T.switch(T.gt(alpha_e, L - 1), L - 1, alpha_e) - T.switch(T.gt(alpha_b, 0), alpha_b, 0))+1])
    #             alpha_t = theano.ifelse.ifelse(T.gt(alpha_b, 0), T.concatenate([T.zeros_like(temp_vector[:alpha_b]), alpha_v]), alpha_v)
    #             alpha_t = theano.ifelse.ifelse(T.ge(alpha_e, L - 1), alpha_t, T.concatenate([alpha_t,T.zeros_like(temp_vector[:L-1-alpha_e])]))
    #             alpha_t = theano.scan(set_alpha_value,
    #                                   sequences=[T.arange(alpha_t.shape[0])],
    #                                   non_sequences=[alpha_t,alpha_pre,t,o,y])
    #             return alpha_t
    #         alphas,_ = theano.scan(to_T,sequences=[T.arange(1,T_ctc)],
    #                                outputs_info = [alpha0],
    #                                non_sequences = [o,y,T_ctc,L])
    #         loss = alphas[-1][-1]+alphas[-1][-2]
    #         loss = T.switch(T.le(loss, 1e-40), 1e-40, loss)
    #         loss = -T.log(loss)
    #         return loss
    #
    #     CTC_LOSSs,_ = theano.scan(each_loss,
    #                               sequences=[T.arange(self.output_shape[0])],
    #                               non_sequences = [T_ctc,L])
    #     self.ctc_loss = theano.function([self.X,self.Y],CTC_LOSSs)
    #     return CTC_LOSSs
    def CTC_LOSS(self):
        outpts = self.output
        inpts = self.Y
        def each_loss(outpt, inpt):
            # y 是填充了blank之后的ans
            blank = 0
            y_nblank = T.neq(inpt, blank)
            n = T.dot(y_nblank, y_nblank)  # 真实的字符长度
            N = 2 * n + 1  # 填充后的字符长度，去除尾部多余的填充
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
            liklihood = fwd_pbblts[-1, -1] + fwd_pbblts[-1, -2]
            liklihood = T.switch(T.lt(liklihood, 1e-32), 1e-32, liklihood)
            loss = -T.log(liklihood)
            return loss
            # return N

        ctc_losss, _ = theano.scan(each_loss,
                                   sequences=[outpts, inpts],
                                   )
        self.ctc_loss = theano.function([self.X, self.Y], ctc_losss)
        return ctc_losss
    
    def CTC_train(self):
        CTC_LOSSs = T.cast(T.mean(self.CTC_LOSS(), axis=0), "float32")
        train_data_d = []
        train_data_m = []
        train_data_m_s = [] 
        learning_rate = T.scalar()
        decay = T.scalar()
        for data in self.train_data:
            data_d = T.grad(CTC_LOSSs, data)
            train_data_d.append(data_d)
            data_m_s = theano.shared(np.zeros(data.get_value().shape).astype(np.float32))
            train_data_m_s.append(data_m_s)
            data_m = data_m_s*decay + (1-decay)*data_d**2
            train_data_m.append(data_m)
            
        #self.data_d_print = theano.function([self.X,self.Y],train_data_d[0][0])
        #upd = [(d,d-learning_rate*d_d)for d,d_d in zip(self.train_data,train_data_d)]
        upd = [(d, d-learning_rate*d_d/T.sqrt(d_m+1e-4))for d,d_d,d_m in zip(self.train_data,train_data_d,train_data_m)]
        upd1 = [(d_m_s, decay*d_m_s+(1-decay)*d_d**2) for d_m_s,d_d in zip(train_data_m_s,train_data_d)]
        upd +=upd1    
        #self.test = theano.function([self.X,self.Y],train_data_d[0])
        self.sgd_train = theano.function([self.X, self.Y, learning_rate, decay],
                                         [],
                                         updates=upd
                                         )
    
    def general_train(self):
        loss = T.sum((self.output - self.Y)**2,axis = 1)
        loss = T.mean(loss,axis = 0)
        self.get_loss = theano.function([self.X,self.Y],loss)
        self.compare_print = theano.function([self.X,self.Y],[self.output[0][:5],self.Y[0][:5],self.X[0][:5]])  
        train_data_d = []
        train_data_m = []
        train_data_m_s = [] 
        learning_rate = T.scalar()
        decay = T.scalar()
        for data in self.train_data:
            data_d = T.grad(loss,data)
            train_data_d.append(data_d)
            data_m_s = theano.shared(np.zeros(data.get_value().shape).astype(np.float32))
            train_data_m_s.append(data_m_s)
            data_m =data_m_s*decay + (1-decay)*data_d**2
            train_data_m.append(data_m)
            
        #self.data_d_print = theano.function([self.X,self.Y],train_data_d[0][0])
        #upd = [(d,d-learning_rate*d_d)for d,d_d in zip(self.train_data,train_data_d)]
        upd = [(d,d-learning_rate*d_d/T.sqrt(d_m+1e-6))for d,d_d,d_m in zip(self.train_data,train_data_d,train_data_m)]
        upd1 = [(d_m_s,decay*d_m_s+(1-decay)*d_d**2) for d_m_s,d_d in zip(train_data_m_s,train_data_d)]
        upd +=upd1    
        #self.test = theano.function([self.X,self.Y],train_data_d[0])
        self.sgd_train = theano.function([self.X, self.Y, learning_rate, decay],
                                         [],
                                         updates = upd
                                         )
            
            
            