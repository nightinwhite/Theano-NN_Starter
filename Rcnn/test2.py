#coding:utf-8
'''
Created on 2016年7月5日

@author: Night
'''
import numpy as np
import theano.tensor as T
import theano
import Get_Date
A = T.vector()
B = T.nnet.softmax(A)
f = theano.function([A], B)
C = B[0][0]
f1 = theano.function([A], T.grad(C, A))
a = [0.0370344,0.0370344,0.0370344,0.0370344,0.0370344,0.0370344]
print f(a)
print f1(a)

