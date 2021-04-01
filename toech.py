# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 15:19:23 2021

@author: Work
"""
#%%imports

import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from impscript import getData

#%%get data 

xData , xLabel = getData('./cifar-10-batches-py/train')
xData = np.reshape(xData , (50000 , 3 , 32,32))
#%%NN
class Net(nn.Module):
    def __init__(self):
        super(Net , self).__init__()
        self.conv1 = nn.Conv2d(3,6,3)
        self.conv2 = nn.Conv2d(6,16, 3)
        
        self.fc1 = nn.Linear(16*6*6,120)
        self.fc2 = nn.Linear(120 , 84)
        self.fc3 = nn.Linear(84 , 10)
    def forward(self , x):
        x = F.max_pool2d(F.relu(self.conv1(x)) , (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)) , (2,2))
        x = x.view(-1 ,self.flatten(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def flatten(self , x ):
        size = x.size()[1:]
        y = 1
        for s in size:
            y *= s
        return y
    
net = Net()
picture = torch.rand(4,3,32,32)
out = net(picture)

print(out)
#%% trial
x = torch.rand(5,requires_grad=True)
y = torch.rand(5 , requires_grad=True)
q = 2*x**3 + y**2
e = torch.tensor([1. , 1. , 1. , 1. , 1.])
q.backward(gradient= e)
print(y.grad)
print(y)
print(x.grad)
print(x)