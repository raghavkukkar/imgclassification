# # -*- coding: utf-8 -*-
# """
# Created on Thu Feb 11 14:27:35 2021

# @author: Raghav
# """
# #weights with dimension classes + bias * pixels + 1
# #score is just matrix multiplication 
import numpy as np
from impscript import getData


class Svm:
    def __init__(self,data,labels ):
        self.data = data/255
        self.data = np.append(self.data , np.ones((self.data.shape[0],1)),axis = 1) #adding a row of ones for addition of bias during matmul
        self.labels = labels
        self.weights = np.random.rand(self.data.shape[1],10) #these weights include bias column
    def lng(self  ,x):
        #loss calculation starts here. Tried without using any loops might be incorrect or could be correct and still slower than loops
        scores = np.matmul(self.data,self.weights) #calculating scores 
        optimal = scores[np.arange(self.data.shape[0]),self.labels] #getting the score of class each image belongs to 
        l = scores - np.matrix(optimal).T + 1 #substracting every score with the class score and adding delta for margin
        l[np.arange(self.data.shape[0]),self.labels] = 0
        l[l<0]= 0
        loss = np.sum(l)/self.data.shape[0]  #calculating aggregate loss for these weights
        regLoss = loss + x*np.sum(np.multiply(self.weights,self.weights)) #l2 regularized loss
        l[l>0] = 1
        count = np.sum(l,axis = 1)
        l[np.arange(self.data.shape[0]),self.labels] = -count.T
        dw = np.matmul(self.data.T,l)
        dw /= self.data.shape[0]
        dw += x*self.weights
        #loss calculations ends here
        return dw , regLoss     
    def train(self ,stepSize,x):
        dw , loss = self.lng(x)
        i = 0
        while i < 1000:
            self.weights += -(stepSize*dw)
            dw , loss = self.lng(x)
            i +=1
            print(i)
        return loss
    def predict(self,yData):
        self.y = yData
        self.y = self.y/255
        self.y = np.append(self.y,np.ones((self.y.shape[0],1)),axis = 1)
        scores = np.matmul(self.y,self.weights)
        plabels = np.argmax(scores , axis = 1)
        return plabels


xData , xLabels = getData('./cifar-10-batches-py/train')
yData, yLabels = getData('./cifar-10-batches-py/test')
sv = Svm(xData[0:25000,:] , xLabels[0:25000])
loss = sv.train(0.01,0.01)
plabels = sv.predict(yData[0:5000,:])

acc = np.mean(plabels == yLabels[0:5000])
