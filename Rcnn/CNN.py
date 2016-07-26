#coding:utf-8
'''
Created on 2015年12月24日

@author: Night
'''
import numpy
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers.core import Dropout,Flatten,Dense,Activation
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
import date
import gzip,cPickle
import keras.layers.LSTM
#----------------------------------------------------
data,label = date.load_data("d:/FFF",2)
numpy.save("FANGdata1.npy",data)
numpy.save("FANGlabel1.npy",label)
#for value in label :
   # print value
# a = numpy.load("last_train_Label3.16.npy")
# for value in a :
#     print value
#
# va_label = date.load_data("d:/last_validate",2500)
# numpy.save("last_validate_Label3.16.npy",va_label)
#----------------------------------------------------
# f = gzip.open('E:/Download/again/mnist.pkl.gz', 'rb')
# (bdata,label),(bva_data,va_label) = cPickle.load(f)
# f.close()
# data = numpy.empty((bdata.shape[0],1,bdata.shape[1],bdata.shape[2]),dtype="int")
# data[:,0,:,:] = bdata[:,:,:]
# va_data = numpy.empty((bva_data.shape[0],1,bva_data.shape[1],bva_data.shape[2]),dtype="int")
# va_data[:,0,:,:] = bva_data[:,:,:]
#----------------------------------------------------
# print data.shape
# print label.shape
# print va_data.shape
# print va_label.shape
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#label = np_utils.to_categorical(label,10)
# va_label = np_utils.to_categorical(va_label,10)
# img_channel = 1
# img_width_c = 40
# img_height_r = 60
#
# model = Sequential()
# model.add(Convolution2D(32,4,4,border_mode='valid',input_shape = (img_channel,img_height_r,img_width_c)))
# model.add(LeakyReLU(alpha=0.3))
# model.add(Convolution2D(32,4,4))
# model.add(LeakyReLU(alpha=0.3))
# model.add(MaxPooling2D())
# model.add(Dropout(0.2))
# model.add(Convolution2D(32,4,4))
# model.add(LeakyReLU(alpha=0.3))
# model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Flatten())
# model.add(Dense(256))
# model.add(LeakyReLU(alpha=0.3))
# model.add(Dropout(0.5))
model.add(Dense(10))
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop',class_mode='categorical')
# #model.load_weights("d:/CNN_Weight_yandx.h5")
model.fit(data, label, batch_size=200, nb_epoch=30,show_accuracy=True, verbose=1, shuffle=True,validation_data=(va_data,va_label))
# model.save_weights("d:/CNN_Weight_yandx_3_11.h5", overwrite=True)
# #model.load_weights("d:/CNN_Weight.h5")
# #answer = model.evaluate(va_data , va_label, 120, show_accuracy = True, verbose=1)
# #print answer