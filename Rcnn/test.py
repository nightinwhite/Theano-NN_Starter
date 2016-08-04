#coding:utf-8
from theano import tensor as T
import theano
import numpy as np
import random
from theano.tensor.shared_randomstreams import RandomStreams
A = T.vector()
seed = np.random.randint(10e6)
rng = RandomStreams(seed=seed)

f = theano.function([], rng.binomial((1,), p=0.5))
f2 = theano.function([], rng.binomial((1,), p=0.5))
print f()
print f2()