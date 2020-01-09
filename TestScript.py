# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:12:23 2019

@author: pille
"""
import scipy.io
import NeuralNetwork
import pandas as pd
import numpy as np
from scipy import stats
   
data = pd.read_excel('.\HTRU_2.xlsx',header=None)
X = data.iloc[:,:8].values
y = data.iloc[:,8:].values.ravel()

X=scipy.stats.zscore(X)
negX=[]
posX=[]
for (x,c) in zip(X,y):
    if(c==0):
        negX.append(x)
    else:
        posX.append(x)
negX=np.array(negX) 
posX=np.array(posX)   

balancedX=[]
balancedy=[]
for i in range(0,1500):
    balancedX.append(posX[i])
    balancedy.append(1)
for i in range(0,1500):
    balancedX.append(negX[i])
    balancedy.append(0)     
balancedX = np.array(balancedX)
balancedy = np.array(balancedy)    

#NN = NeuralNetwork.NeuralNetwork([[1],[2],[3]],[0,0,0]) 
#NN.loadPulsar()
#k=NN.addLayer(10,"relu")
#NN.finalLayer()
#test = X[0]
#print(NN.predict(test))

       

