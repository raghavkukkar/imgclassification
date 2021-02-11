# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:31:14 2021

@author: Raghav
"""
import numpy as np
def NN(x,y,xlabels):
    ypred = np.zeros(y.shape[0],dtype=xlabels.dtype)
    for i in range(y.shape[0]):
        d= np.sum(np.abs(x - y[i,:]),axis=1)
        ypred[i] = xlabels[np.argmin(d)]
    return ypred

def KNN(x,y,xlabels,k):
    ypred = np.zeros(y.shape[0],dtype=xlabels.dtype)
    for i in range(y.shape[0]):
        d= np.sum(np.abs(x - y[i,:]),axis=1)
        close = xlabels[np.argsort(d)[0:k]]       
        ypred[i] = np.argmax(np.bincount(close))
    return ypred