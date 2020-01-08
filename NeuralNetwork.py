# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 00:22:16 2020

@author: pille
"""
import pandas as pd

class NeuralNetwork:
    def __init__(self,X=None,y=None):
        self.X=X
        self.y=y
        
    def loadPulsar(self):
        data = pd.read_excel('.\HTRU_2.xlsx',header=None)
        self.X = data.iloc[:,:8].values
        self.y = data.iloc[:,8:].values.ravel()
     
    def showData(self):
        print('X: ',self.X,'\ny: ',self.y)
        
    def hello(self):
        print('Hey Chiara :D')