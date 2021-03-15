# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 16:15:34 2021

@author: Raghav
"""

import tensorflow.keras as keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

#%% Make model here

input_tensor=Input((3072))
dns1=Dense(units=100)(input_tensor)
act1=Activation(keras.activations.relu)(dns1)
dns2=Dense(units=50)(act1)
act2=Activation(keras.activations.relu)(dns2)
dns3 = Dense(units = 10)(act2)
act3 = Activation(keras.activations.softmax)(dns3)

model=Model(inputs=input_tensor, outputs=act3)

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001), loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])
model.summary()
#%% load data X_test
from impscript import getData
X_train , Y_train= getData('./cifar-10-batches-py/train')
X_test, Y_test= getData('./cifar-10-batches-py/test')

#%%One hot encoding
from tensorflow.keras.utils import to_categorical
Y_train=to_categorical(Y_train)
Y_test=to_categorical(Y_test)

#%%

model.fit(X_train[0:25000]/255, Y_train[0:25000], epochs=50, batch_size=100, validation_data=(X_test/255, Y_test))