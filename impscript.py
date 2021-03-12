# trying some numpy to learn 

##########################
#import numpy 
import numpy as np
import pickle
import time
from os import listdir
from os.path import isfile, join


def getData(src):
    imgs = []
    labels = []
    paths = [join(src,f) for f in listdir(src) if isfile(join(src,f))]
    for f in paths:
        with open(f,'rb') as file:
            x = pickle.load(file,encoding="latin1")
            img = x['data']
            label = x['labels']
            imgs.append(img)
            labels.append(np.array(label))
    return np.concatenate(imgs).reshape(len(imgs)*imgs[0].shape[0],3072).astype(np.float64), np.concatenate(labels).astype(np.uint8)
            

def imageMean(X):
    return np.mean(X)

# xData , xLabels = getData('./cifar-10-batches-py/train')
# yData, yLabels = getData('./cifar-10-batches-py/test')
# t0 = time.time()
# prdLabels = KNN(xData,yData[0:100,:],xLabels,7)
# runTime = time.time() - t0
# accuracy = np.mean( prdLabels == yLabels[0:100]) 