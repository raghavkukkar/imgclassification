import numpy as np
from impscript import getData

# DESIGN CONSIDERATION COULD HAVE IMPLEMENTED IT DIFFERENTLY
class Activations:
    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))
class Backgates:
    @staticmethod
    def sigPrime(x,y =1):
        return y*Activations.sigmoid(x)*(1 - Activations.sigmoid(x))

# MAIN CLASS FOR NETWORK 

# WORK TO DO - GENERALIZE THE BACKPROP ALGO
class Network:
    def __init__(self,size,_input,_tags , _validation ,_validtags,reg ):
        self.input = _input/255
        self.valid = _validation/255
        self.vtags = _validtags
        self.tags = _tags
        self.num_layers = len(size) - 1
        self.sizes = size
        self.bias = [0.01*np.random.randn(y,1) for y in (self.sizes[1:])]

        self.weights = [0.01*np.random.randn(y,x) for x , y in zip(self.sizes[:-1],self.sizes[1:])]
        
        self.gdw = []
        self.gdb = []
        self.convTags()
    
    def convTags(self):
        x = np.zeros((self.sizes[-1] , self.tags.shape[0]))
        x[(self.tags),np.arange(40000)] = 1
        self.tags = x
        
    def forwardPass(self):
        a = self.valid
        
        for b,w in zip(self.bias , self.weights):
            a = Activations.sigmoid(np.matmul(w,a) + b)
        return a

# SOME BAD SHIT IS HAPPENING HERE DEBUGGING ...
    def update(self,rate):
        for i in range(self.num_layers):
            self.bias[i] = self.bias[i] - (rate*self.gdb.pop())
            
            self.weights[i] = self.weights[i] - rate*self.gdw.pop()
    
     
    def accuracy(self): 
        y=  self.forwardPass().argmax(axis = 0 )
        return np.mean(self.vtags == y)
        
    def train(self , rate,epochs , groups):
      
        
        for e in range(epochs):
            print(e)
            for t in range(0,self.input.shape[0] , groups):
                
                    activate = [self.input[:,t: t + groups]]
                    zs = []
                    
                    #forward pass
                    for x , y in zip(self.weights,self.bias):
                        zs.append(np.matmul(x,activate[-1]) + y)
                        activate.append(Activations.sigmoid(zs[-1]))
        
                    #backward pass
                    x = Backgates.sigPrime(zs[-1],activate[-1] - self.tags[:,t: t + groups] )
                    self.gdb.append(x.sum(axis = 1,keepdims = True))
                    
                    self.gdw.append(np.matmul(x,activate[-2].T))
                    x = np.matmul(self.weights[-1].T , x)
                    x = Backgates.sigPrime(zs[-2],x)
                    self.gdb.append(x.sum(axis = 1 , keepdims = True))
                    self.gdw.append(np.matmul(x,activate[-3].T))
                    self.update(rate)



data , labels = getData('./cifar-10-batches-py/train')
data= data.T

nn = Network(((32*32*3),100,10), data[:,0:40000], labels[0:40000], data[:,40000:50000], labels[40000:50000])

nn.train(0.01,90,50)

acc= nn.accuracy()

