# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 00:22:16 2020

@author: pille
"""
import pandas as pd
import numpy as np


class NeuralNetwork:
    ActivationFunctions={ "relu" : (lambda x: max(0,x)), "id":(lambda x:x) }
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
        self.layers.append((np.random.rand(self.layers[-1][0].shape[1],d),NeuralNetwork.ActivationFunctions.get("id")))
        
    def predict(self,X):
        X = np.array(X)
        X = output(X)
        return np.where(X==np.amax(X))[0]    
        
    def output(self,x):
        for l in self.layers:
            x = x.dot(l[0])
        return x    
        
    def fit(self):
        for x,c in X,y:
            for l in self.layers:
                x = x.dot(l[0])
                
                
    def loadPulsar(self):
        self.layers = []
        data = pd.read_excel('.\HTRU_2.xlsx',header=None)
        self.X = data.iloc[:,:8].values
        self.y = data.iloc[:,8:].values.ravel()
     
    def showData(self):
        print('X: ',self.X,'\ny: ',self.y)
        
    def hello(self):
        print('Hey Chiara :D')