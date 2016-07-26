#coding:utf-8
from theano import tensor as T
import theano
import numpy as np
import random

A = T.tensor3()
B = T.matrix()
f = theano.function([A, B], T.dot(B, A))
a = np.arange(3*5*6).reshape(3, 6, 5).astype(np.float64)
b = np.asarray([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
#b = np.transpose(b, [1,0])
c = f(a, b)
print c
#c = np.transpose(b, [1,0])
