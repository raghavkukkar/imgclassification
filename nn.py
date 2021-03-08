#ULTIMATE TODO PRACTISE INTEGRAL TO ROMAN NUMERAL AND ONE MORE MEDIUM LEETCODE PROBLEM


import numpy as np
from impscript import getData

# DESIGN CONSIDERATION COULD HAVE IMPLEMENTED IT DIFFERENTLY AND YES IT LOOKS BAD BUT I WANTED TO STATIC METHODS AND CLASSES, BECAUSE IT JUST 
# MAKES THE STRUCTURE OF CODE CLEAN FOR ME SO THIS IS HOW IT IS IMPLEMENTED BYE! 

class Activations:
    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))
    @staticmethod
    def relu(x):
       
        return np.maximum(0,x)
    @staticmethod
    def softmax(x):
        y = np.exp(x)
        return y/np.sum(y,axis = 0)
        
#TODO SOFTMAX ACTIVATION BACKGATE - I'M NOTGOING TO IMPLEMENT TODAY BECAUSE I COULD NOT REALLY UNDERSTAND THE DERIVATIVE.
class Backgates:
    @staticmethod
    def sigPrime(x,y =1):
        return y*Activations.sigmoid(x)*(1 - Activations.sigmoid(x))
    @staticmethod 
    def reluPrime(x,y = 1):
        x[x<= 0 ] = 0
        return y*(x)
    
# MAIN CLASS FOR NETWORK 

# WORK TO DO - GENERALIZE THE BACKPROP ALGO AND ADD CROSS ENTROPY LOSS CALCULATION AND TRY IT WITH SOFTMAX RELU SIGMOID
#TODO CONTINUE - ADD MINI BATCH RANDOMIZATION AND TRY IF WE GET BETTER RESULTS
class Network:
    def __init__(self,size,_input,_tags , _validation ,_validtags,reg ,methods ):
        self.input = _input/255
        self.reg = reg
        self.valid = _validation/255
        self.vtags = _validtags
        self.tags = _tags
        self.num_layers = len(size) - 1
        self.sizes = size
        self.bias = [0.001*np.random.randn(y,1) for y in (self.sizes[1:])]
        self.methods = methods
        self.weights = [0.001*np.random.randn(y,x) for x , y in zip(self.sizes[:-1],self.sizes[1:])]
        
        self.gdw = []
        self.gdb = []
        
    
    def convTags(self):
        x = np.zeros((self.sizes[-1] , self.tags.shape[0]))
        x[(self.tags),np.arange(40000)] = 1
        self.tags = x
        
    def forwardPass(self):
        a = self.valid
        for b,w,m in zip(self.bias , self.weights,self.methods):
            a = m(np.matmul(w,a) + b)
        return a


    def update(self,rate):
        for i in range(self.num_layers):
            self.bias[i] += -(rate*self.gdb.pop())
            self.weights[i] += -(rate*self.reg*self.gdw.pop())
    
     
    def accuracy(self): 
        y=  self.forwardPass().argmax(axis = 0 )
        
        return np.mean(y == self.vtags)
        
    def train(self , rate,epochs , groups ):
      
        
        for e in range(epochs):
            print(e)
            for t in range(0,self.input.shape[0] , groups):
                
                    activate = [self.input[:,t: t + groups]]
                    zs = []
                    
                    #FORWARD PASS OR FORWARD PASS
                    for x , y , m in zip(self.weights,self.bias,self.methods):
                        zs.append(np.matmul(x,activate[-1]) + y)
                        activate.append(m(zs[-1]))
        
                    #BACKWARD PASS OR BACKPROP
                    x = activate[-1]
                    x[self.tags[t:t+groups],np.arange(0,groups)] -=1 #might be wrong .... most probably wrong NEED CHANGES AND RESEARCH
                    self.gdb.append(x.sum(axis = 1,keepdims = True))
                    self.gdw.append(np.matmul(x,activate[-2].T))
                    x = np.matmul(self.weights[-1].T , x)
                    x = Backgates.reluPrime(zs[-2],x)
                    self.gdb.append(x.sum(axis = 1 , keepdims = True))
                    self.gdw.append(np.matmul(x,activate[-3].T))
                    self.update(rate)



data , labels = getData('./cifar-10-batches-py/train')
data= data.T

nn = Network(((32*32*3),50,10), data[:,0:40000], labels[0:40000], data[:,40000:50000], labels[40000:50000],0.5,(Activations.relu,Activations.softmax))

nn.train(0.0001,2000,100)

acc= nn.accuracy()

