# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:27:35 2021

@author: Raghav
"""
#weights with dimension classes + bias * pixels + 1
#score is just matrix multiplication 
import numpy as np
from impscript import getData


class SVM(object):
    def __int__(self, data , labels , weights):
        self.data = np.append(data , np.ones(1,data.shape[1]),axis = 0) #adding a row of ones for addition of bias during matmul
        self.labels = labels
        self.weights = weights #these weights include bias column
    def train(self , stepSize,x):
        #loss calculation starts here. Tried without using any loops might be incorrect or could be correct and still slower than loops
        dw = np.zeros_like(self.weights)
        scores = np.matmul(self.weights,self.data) #calculating scores 
        optimal = scores[self.labels,np.arange(self.data.shape[1])] #getting the score of class each image belongs to 
        l = scores - optimal + 1 #substracting every score with the class score and adding delta for margin
        loss = np.sum(l)  #calculating aggregate loss for these weights
        regLoss = loss/self.data.shape[1] + x*np.sum(self.weights*self.weights) - 1 #l2 regularized loss
        # dW[:,y[i]] -= (l[l>0].shape[0] -1)*self.data  
        # dW[:,j] += X[i,:]
        #loss calculations ends here
             
    def predict(self,yData):
        pass

xData , xLabels = getData('./cifar-10-batches-py/train')
yData, yLabels = getData('./cifar-10-batches-py/test')
