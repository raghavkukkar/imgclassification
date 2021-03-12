#ULTIMATE TODO PRACTISE INTEGRAL TO ROMAN NUMERAL AND ONE MORE MEDIUM LEETCODE PROBLEM
#IDEA FOR GENERALIZING(SPELL CHECK) THE BACKPROP: THE FIRST THING WE ARE DOING IS GETTING THE LOSS OUTPUT SO IN I = 0 WE ASSIGN IT TO A VARIABLE
#THEN WE PASS IT THROUGH LAST TWO GATES THAT COULD BE COMBINED LIKE IN THE CASE OF SOFTMAX AND CROSS ENTROPY.
#THEN WE ARE APPENDING 
#THEN WE PROPEGATE BY MATMUL SO IF I != 0 WE MATMUL AND THEN ACTIVATE AND THEN APPEND SO THEIR IS A LOOP 

import numpy as np
from impscript import getData
import time

# DESIGN CONSIDERATION COULD HAVE IMPLEMENTED IT DIFFERENTLY AND YES IT LOOKS BAD BUT I WANTED TO STATIC METHODS AND CLASSES, BECAUSE IT JUST 
# MAKES THE STRUCTURE OF CODE CLEAN FOR ME SO THIS IS HOW IT IS IMPLEMENTED BYE! 


class Activations:
    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))
    @staticmethod
    def relu(x):
        t = np.maximum(0 , x)
        return t
    @staticmethod 
    def leakyRelu(x):
        t = np.copy(x)
        t[t < 0] = 0.01*t[t < 0]
        return t
        
    @staticmethod
    def softmax(x):
        y = np.exp(x)
        return y/np.sum(y,axis = 0)
        


#TODO SOFTMAX ACTIVATION BACKGATE - I'M NOTGOING TO IMPLEMENT TODAY BECAUSE I COULD NOT REALLY UNDERSTAND THE DERIVATIVE.
#UPDATE I UNDERSTOOD AND KINDA IMPLEMENTED THE LOSS + SOFTMAX GATE 
class Backgates:
    @staticmethod
    def sigPrime(x,y =1):
        return y*Activations.sigmoid(x)*(1 - Activations.sigmoid(x))
    @staticmethod 
    def reluPrime(x,y ):
        y[x <= 0] = 0
        
    @staticmethod
    def leakyReluPrime(x, y = 1):
        x[x > 0] = 1
        x[x< 0] = 0.01
        return x
    
# MAIN CLASS FOR NETWORK 

# WORK TO DO - GENERALIZE THE BACKPROP ALGO
#TODO CONTINUE - ADD MINI BATCH RANDOMIZATION AND TRY IF WE GET BETTER RESULTS
class Network:
    def __init__(self,size,_input,_tags , _validation ,_validtags,reg ,methods ):
        self.input = _input/255.0
        self.reg = reg
        self.valid = _validation/255.0
        self.vtags = _validtags
        self.tags = _tags
        self.num_layers = len(size) - 1
        self.sizes = size
        self.bias = [np.zeros((y,1)) for y in (self.sizes[1:])]
        self.methods = methods
        self.weights = [0.01*np.random.randn(y,x) for x , y in zip(self.sizes[:-1],self.sizes[1:])]
        
        self.gdw = []
        self.gdb = []
        print(self.input.shape[1])
    
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
            a = m(np.matmul(w,a) + b)
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
            t1 = time.time()
            for t in range(0,self.input.shape[1] , groups):
                
                    activate = [self.input[:,t: t + groups]]
                    zs = []
                    
                    #FORWARD PASS OR FORWPROP
                    for x , y , m in zip(self.weights,self.bias,self.methods):
                        zs.append(np.matmul(x,activate[-1]) + y)
                        activate.append(m(zs[-1]))
                        
                    d = activate[-1][self.tags[t:t+groups],np.arange(0,groups)]
                    d = -np.log(d)
                    d = np.sum(d)/groups
                    # regLoss = 0.5*self.reg*np.sum(self.weights[0] * self.weights[0])+0.5*self.reg*np.sum(self.weights[1] * self.weights[1])
                    #BACKWARD PASS OR BACKPROP
                    x = activate[-1]
                    x[self.tags[t:t+groups],np.arange(0,groups)] -=1#might be wrong .... most probably wrong NEED CHANGES AND RESEARCH
                    x /= groups
                    self.gdb.append(x.sum(axis = 1,keepdims = True))
                    
                    self.gdw.append(np.matmul(x,activate[-2].T))
                    
                    x = np.matmul(self.weights[-1].T , x)
                    Backgates.reluPrime(zs[-2],x)
                    self.gdb.append(x.sum(axis = 1 , keepdims = True))
                    
                    self.gdw.append(np.matmul(x,activate[-3].T))
                    
                    self.update(rate)
            print(time.time() - t1)
                    
        return d

data , labels = getData('./cifar-10-batches-py/train')
data= data.T
yData, yLabels = getData('./cifar-10-batches-py/test')
yData = yData.T
nn = Network(((32*32*3),50,10), data[:,0:25000], labels[0:25000], data[:,40000:50000], labels[40000:50000],0.01,(Activations.relu,Activations.softmax))

loss = nn.train(0.001,50,100)

# accValidation = nn.accuracy()

# accTrain = nn.accuracy(yData , yLabels)

