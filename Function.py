# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 10:50:58 2021

@author: Raghav
"""
import numpy as np


class Sigmoid:
    @staticmethod
    def activation(x):
        return 1.0/(1.0 + np.exp(-x))
    @staticmethod
    def derivative(x,y =1):
        return y*Sigmoid.activation(x)*(1 - Sigmoid.activation(x))



class Relu:
    @staticmethod
    def activation(x):
        t = np.maximum(0 , x)
        return t
    @staticmethod
    def derivative(x,y):
        y[x <= 0] = 0



class LeakyRelu:
    @staticmethod 
    def activation(x):
        t = np.copy(x)
        t[t < 0] = 0.01*t[t < 0]
        return t
        
    @staticmethod
    def derivative(x, y):
        y[x <= 0] = 0
    
    
    
class Softmax:
    @staticmethod
    def activation(x):
        y = np.exp(x)
        return y/np.sum(y,axis = 0)


class Cost:
    @staticmethod
    def crossEntropy(activate ,target):
        d = activate[target,np.arange(0,activate.shape[1])]
        d = -np.log(d)
        d = np.sum(d)/activate.shape[1]
        return d