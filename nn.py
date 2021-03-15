# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 10:50:58 2021

@author: Raghav
"""

import numpy as np
from impscript import getData
from Function import Relu , Softmax , Cost
    
# MAIN CLASS FOR NETWORK 


class Network:
    
    def __init__(self,size,_input,_tags , _validation ,_validtags):
        self.input = _input/255
        self.valid = _validation/255
        self.vtags = _validtags
        self.tags = _tags
        self.num_layers = len(size) - 1
        self.sizes = size
        self.bias = [np.zeros((y,1)) for y in (self.sizes[1:])]
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for x , y in zip(self.sizes[:-1],self.sizes[1:])]
        self.gdw = []
        self.gdb = []
    
    def setParams(self ,reg ,  methods , costFunction):
        
        self.reg = reg
        self.methods = methods
        self.costFunction = costFunction
    
    def convTags(self):
        x = np.zeros((self.sizes[-1] , self.tags.shape[0]))
        x[(self.tags),np.arange(40000)] = 1
        self.tags = x
        
    def forwardPass(self, x = None):
        if x is None:
            a = self.valid
        else:
            a = x 
        for b,w,m in zip(self.bias , self.weights,self.methods):
            a = m.activation(np.matmul(w,a) + b)
        return a

    def update(self,rate):
        for i in range(self.num_layers):
            self.bias[i] += -(rate*self.gdb.pop())
            self.weights[i] += -rate*(self.gdw.pop() + self.reg*self.weights[i])
         
    def accuracy(self , x = None, xtags = None):
        if( x is None or xtags is None):    
            y=  self.forwardPass().argmax(axis = 0 )
            return np.mean(y == self.vtags)
        else:
            y = self.forwardPass(x).argmax(axis = 0)
            return np.mean(y == xtags)
                
    def train(self , rate,epochs , groups ):      
        
        for e in range(epochs):
            print(e, "yo oy")
    
            for t in range(0,self.input.shape[1] , groups):
                
                    activate = [self.input[:,t: t + groups]]
                    zs = []
                    
                    #FORWARD PASS OR FORWPROP
                    for x , y , m in zip(self.weights,self.bias,self.methods):
                        zs.append(np.matmul(x,activate[-1]) + y)
                        activate.append(m.activation(zs[-1]))
                    
                    d = self.costFunction(activate[-1] , self.tags[t:t+groups])    
                    print(d)
                    # regLoss = 0.5*self.reg*np.sum(self.weights[0] * self.weights[0])+0.5*self.reg*np.sum(self.weights[1] * self.weights[1])
                   
                    #BACKWARD PASS OR BACKPROP
                    x = activate[-1]
                    x[self.tags[t:t+groups],np.arange(0,groups)] -=1
                    x /= groups
                    self.gdb.append(x.sum(axis = 1,keepdims = True))
                    self.gdw.append(np.matmul(x,activate[-2].T))
                    for i in range(2 , self.num_layers + 1):
                        x = np.matmul(self.weights[-(i-1)].T , x)
                        self.methods[-i].derivative(zs[-i],x)
                        self.gdb.append(x.sum(axis = 1 , keepdims = True))    
                        self.gdw.append(np.matmul(x,activate[-(i+1)].T))
                    self.update(rate)                    
        return d

data , labels = getData('./cifar-10-batches-py/train')
data= data.T
yData, yLabels = getData('./cifar-10-batches-py/test')
yData = yData.T
nn = Network(((32*32*3),100,50,10), data[:,0:25000], labels[0:25000], data[:,40000:50000], labels[40000:50000])
nn.setParams(0.01, (Relu , Relu , Softmax) , Cost.crossEntropy)
loss = nn.train(0.001,50,100)

# accValidation = nn.accuracy()

# accTrain = nn.accuracy(yData , yLabels)

