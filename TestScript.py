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
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import tree
   
data = pd.read_excel('.\HTRU_2.xlsx',header=None)
X = data.iloc[:,:8].values
y = data.iloc[:,8:].values.ravel()

X=scipy.stats.zscore(X)
negX=np.array([X[i] for i in range (0,len(X)) if y[i]==0])
posX=np.array([X[i] for i in range (0,len(X)) if y[i]==1])

trainX= np.concatenate((posX,negX[sample_without_replacement(len(negX),len(posX))]),axis=0)
trainy= np.array([1 for x in posX]+[0 for x in posX] )  

acs=["identity", "logistic", "tanh", "relu"]
sls=[2,5,10,20,50,100]
kf = KFold(n_splits=10,shuffle=True)
best = 0
bestt = None
bestlist=[]
"""
for act in acs:
    for nl in range(1,10):
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
sum1=0 
sum2=0
predy = [0 for x in trainy]
predtreey = [0 for x in trainy]
for train_index,test_index in kf.split(trainX):
    innerTrainX=trainX[train_index]
    innerTrainy=trainy[train_index]
    bestNN = None
    bestTree= None
    besttreeScore=0
    bestScore = 0
    for train_index_inner,test_index_inner in kf.split(innerTrainX):
        NN = MLPClassifier(max_iter=1000,activation="logistic", hidden_layer_sizes=(10,10,10,10,10))
        treetwo = tree.DecisionTreeClassifier(max_depth=4,criterion='gini')
        NN.fit(innerTrainX[train_index_inner],innerTrainy[train_index_inner])
        treetwo.fit(innerTrainX[train_index_inner],innerTrainy[train_index_inner])
        if NN.score(innerTrainX[test_index_inner],innerTrainy[test_index_inner])>bestScore:
            bestNN=NN
            bestScore=NN.score(innerTrainX[test_index_inner],innerTrainy[test_index_inner])
        if treetwo.score(innerTrainX[test_index_inner],innerTrainy[test_index_inner])>besttreeScore:
            bestTree=treetwo
            bestScore=treetwo.score(innerTrainX[test_index_inner],innerTrainy[test_index_inner])
    sum1 += bestNN.score(trainX[test_index],trainy[test_index])
    sum2 += bestTree.score(trainX[test_index],trainy[test_index])
    for i,j in zip(test_index,bestNN.predict_proba(trainX[test_index])[:,1:].ravel()):
        predy[i] = j
    for i,j in zip(test_index,bestTree.predict_proba(trainX[test_index])[:,1:].ravel()):
        predtreey[i] = j    
    
sum1= sum1/10
sum2= sum2/10

print(sum1) 
print(sum2)

fpr1, tpr1, thresholds = metrics.roc_curve(trainy, predy)
roc_auc1 = metrics.auc(fpr1, tpr1)

fpr2, tpr2, thresholds = metrics.roc_curve(trainy, predtreey)
roc_auc2 = metrics.auc(fpr2, tpr2)

plt.plot(fpr1, tpr1, 'b-', label = ("ANN (AUC "+str(round(roc_auc1,3))+")"))
plt.plot(fpr2, tpr2, 'g-', label = ("Tree (AUC "+str(round(roc_auc2,3))+")"))
plt.plot(range(0,2),range(0,2), '--k', label = "Random")
plt.plot([0,0.1,0.1],[0.75,0.75,1],"r-")
plt.plot([0,0,0.1],[0.75,1,1],"r-")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")    
plt.legend()
plt.savefig("Roc.pdf",dpi=2000,format="pdf")
plt.show()

plt.plot(fpr1, tpr1, 'b-', label = ("ANN (AUC "+str(round(roc_auc1,3))+")"))
plt.plot(fpr2, tpr2, 'g-', label = ("Tree (AUC "+str(round(roc_auc2,3))+")"))
plt.plot([0,0.1,0.1],[0.75,0.75,1],"r-")
plt.plot([0,0,0.1],[0.75,1,1],"r-")
plt.ylim(0.749,1.001)
plt.xlim(-0.0005,0.1005)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")    
plt.legend()
plt.savefig("RocZoom.pdf",dpi=2000,format="pdf")

cm= confusion_matrix(trainy,np.round(predy,0))
print(cm)  
cm= confusion_matrix(trainy,np.round(predtreey,0))
print(cm)      

