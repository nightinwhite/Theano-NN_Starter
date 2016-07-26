from theano import tensor as T
import theano
import numpy as np
import Layer
X = T.matrix()
X_shape = (1,100)
l = Layer.Dense_Layer(20)
l.bulid(X_shape)
l.set_input(X)
O = l.get_output()
f = theano.function([X],O)
x = np.arange(X_shape[0]*X_shape[1]).reshape(X_shape).astype(np.float32)
y = f(x)
print x
print y
print y.shape
print l.get_output_shape()