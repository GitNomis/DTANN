# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 00:22:16 2020

@author: pille
"""
import pandas as pd
import numpy as np
import math


class NeuralNetwork:
    ActivationFunctions={ "relu" : ((lambda x: max(0,x),(lambda x: (x > 0 and 1 or 0)))), "id":((lambda x:x),(lambda x:1)), "sigmoid":((lambda x: 1/(1+math.exp(-x))),(lambda x:(math.exp(-x)/(1+math.exp(-x))**2))) }
    def __init__(self,X,y):
        self.layers = []
        self.X=np.array(X)
        self.y=np.array(y)
        
    def addLayer(self,u,a="relu"):
        if not self.layers:
            d = self.X.shape[1]
        else:
            d = self.layers[-1][0].shape[1]
        layer = np.random.rand(d,u)
        if not a in NeuralNetwork.ActivationFunctions.keys():
            print(str(a)+" is not an implemented activation function. Defaulted to \"relu\"")
            a=NeuralNetwork.ActivationFunctions.get("relu")  
        self.layers.append((layer,NeuralNetwork.ActivationFunctions.get(a)))
        return self.layers
    
    def finalLayer(self):
        d = np.unique(self.y).size
        self.layers.append((np.random.rand(self.layers[-1][0].shape[1],d),NeuralNetwork.ActivationFunctions.get("sigmoid")))
        
    def predict(self,X):
        X = np.array(X)
        for l in self.layers:
            X = self.apply(X.dot(l[0]),l[1][0])
        return X#np.where(X==np.amax(X))[0] 
    
    def fit(self,X=None,y=None):
        if not X:
            X=self.X
            y=self.y
        change = self.nabla(X[0],y[0])
        for i in range(1,y.shape[0]):
            nm = self.nabla(X[i],y[i])
            for m in range(0,len(change)):
                change[m]+=nm[m]
        for i in range(0,len(change)):
            change[i]= change[i]/y.shape[0]
        for i in range(0,len(self.layers)):
            self.layers[i]=(self.layers[i][0]-change[i],self.layers[i][1])
        return self.layers    
            
    def nabla(self,inx,outy):
        expected = [0 for x in range(0,self.layers[-1][0].shape[1]) ]
        expected[outy]=1
        nodes = self.simulate(inx)
        newlayers = []
        lastlayer = []
        for i in range(0,self.layers[-1][0].shape[1]):
            lastlayer.append((2*(nodes[-1][i]-expected[i])))
        for index in range(len(self.layers)-1,-1,-1):
            l = self.layers[index]
            dfrom = l[0].shape[0]
            dto = l[0].shape[1]
            m = np.ndarray(shape=(dfrom,dto))
            for i in range(0,dfrom):
                for j in range(0,dto):
                    m[i,j] = nodes[index][i] *  l[1][1](l[0][i,j]*nodes[index][i]) * lastlayer[j]
            newlayers.insert(0,m)        
            lastnewlayer = []
            for i in range(0,dfrom):
                sum=0
                for j in range(0,dto):
                    sum +=  l[0][i,j] * l[1][1](l[0][i,j]*nodes[index][i]) * lastlayer[j]
                lastnewlayer.append(sum)
            lastlayer = lastnewlayer
        return newlayers    
                    
    def simulate(self,x):
        vectors = [x]
        for l in self.layers:
            x = self.apply(x.dot(l[0]),l[1][0])
            vectors.append(x)   
        return vectors  
    
    def apply(self,x,lam):
        for i in range(0,x.shape[0]):
            x[i]=lam(x[i])
        return x    
                    
    def loadPulsar(self):
        self.layers = []
        data = pd.read_excel('.\HTRU_2.xlsx',header=None)
        self.X = data.iloc[:,:8].values
        self.y = data.iloc[:,8:].values.ravel()
     
    def showData(self):
        print('X: ',self.X,'\ny: ',self.y)
        
    def hello(self):
        print('Hey Chiara :D')
     