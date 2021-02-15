# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:57:32 2021

@author: Work
"""
import numpy as np
from impscript import getData
import time
xData , xLabels = getData('./cifar-10-batches-py/train')
yData, yLabels = getData('./cifar-10-batches-py/test')
xData = xData.T/255
weights = np.random.rand(10,xData.shape[0])
scores = np.matmul(weights,xData)

yi_scores = scores[xLabels,np.arange(scores.shape[1])] 
l = scores - yi_scores + 1


l[xLabels,np.arange(scores.shape[1])] = 0
loss = np.sum(l[l > 0 ])/50000

# loss = np.mean(np.sum(l,axis=1))

l[l > 0] = 1
l[l<0] = 0
count = np.sum(l , axis = 0)
l[xLabels,np.arange(xData.shape[1])] = -count

# dw = np.dot(xData.T,l)

# dw /= xData.shape[0]