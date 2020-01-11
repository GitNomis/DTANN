# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:12:23 2019

@author: pille
"""
import scipy.io
import pandas as pd
import numpy as np
from sklearn.utils.random import sample_without_replacement
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

   
data = pd.read_excel('.\HTRU_2.xlsx',header=None)
X = data.iloc[:,:8].values
y = data.iloc[:,8:].values.ravel()

X=scipy.stats.zscore(X)
negX=np.array([X[i] for i in range (0,len(X)) if y[i]==0])
posX=np.array([X[i] for i in range (0,len(X)) if y[i]==1])

trainX= np.concatenate((posX,negX[sample_without_replacement(len(negX),len(posX))]),axis=0)
trainy= np.array([1 for x in posX]+[0 for x in posX] )  

acs=["logistic"]
sls=[6,7,8,9,10,11,12,13,14,15] #[2,5,10,20,50,100]
kf = KFold(n_splits=10,shuffle=True)
best = 0
bestt = None
bestlist=[]

for act in acs:
    for nl in range(3,10):
        print(act,nl)    
        for sl in sls:
            sum = 0
            for train_index,test_index in kf.split(trainX):
                innerTrainX=trainX[train_index]
                innerTrainy=trainy[train_index]
                bestNN = None
                bestScore = 0
                for train_index_inner,test_index_inner in kf.split(innerTrainX):
                    NN = MLPClassifier(max_iter=2000,activation=act, hidden_layer_sizes=tuple([sl for i in range(0,nl)])) 
                    NN.fit(innerTrainX[train_index_inner],innerTrainy[train_index_inner])
                    if NN.score(innerTrainX[test_index_inner],innerTrainy[test_index_inner])>bestScore:
                        bestNN=NN
                        bestScore=NN.score(innerTrainX[test_index_inner],innerTrainy[test_index_inner])
                sum+= bestNN.score(trainX[test_index],trainy[test_index])
            sum= sum/10
            print(sum,nl,":",sl)
            if sum > best:
                best=sum
                bestt = (act,nl,sl)
                bestlist.append(bestt)
                print(act,"numberLayers:",nl,"layerSizes",sl)
print(bestt) 
print(bestlist)           
"""    
sum=0 
for train_index,test_index in kf.split(trainX):
    innerTrainX=trainX[train_index]
    innerTrainy=trainy[train_index]
    bestNN = None
    bestScore = 0
    for train_index_inner,test_index_inner in kf.split(innerTrainX):
        NN = MLPClassifier(max_iter=1000,activation="logistic", hidden_layer_sizes=(10,10,10,10,10)) 
        NN.fit(innerTrainX[train_index_inner],innerTrainy[train_index_inner])
        if NN.score(innerTrainX[test_index_inner],innerTrainy[test_index_inner])>bestScore:
            bestNN=NN
            bestScore=NN.score(innerTrainX[test_index_inner],innerTrainy[test_index_inner])
              
    print(bestNN.score(trainX[test_index],trainy[test_index]))        
    sum += bestNN.score(trainX[test_index],trainy[test_index])

sum= sum/10
print(sum)            
"""
       

