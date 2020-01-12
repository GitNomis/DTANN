# -- coding: utf-8 --
"""
Created on Sat Jan 11 11:32:45 2020

@author: CC114402
"""

import scipy.io
import pandas as pd
import numpy as np
from sklearn.utils.random import sample_without_replacement
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn import tree
#import treeprint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

classes = ["RFI","Pulsar"]
attributes = ["Mean of the integrated profile","Standard deviation of the integrated profile","Excess kurtosis of the integrated profile","Skewness of the integrated profile","Mean of the DM-SNR curve","Standard deviation of the DM-SNR curve","Excess kurtosis of the DM-SNR curve","Skewness of the DM-SNR curve"]

   
data = pd.read_excel('.\HTRU_2.xlsx',header=None)
X = data.iloc[:,:8].values
y = data.iloc[:,8:].values.ravel()
negX=np.array([X[i] for i in range (0,len(X)) if y[i]==0])
posX=np.array([X[i] for i in range (0,len(X)) if y[i]==1])

trainX= np.concatenate((posX,negX[sample_without_replacement(len(negX),len(posX))]),axis=0)
trainy= np.array([1 for x in posX]+[0 for x in posX] )  

boom = tree.DecisionTreeClassifier(criterion='gini',min_samples_split=100, max_depth=4)
boom = boom.fit(trainX, trainy)
#treeprint.tree_print(boom, attributes, classes)
tree.plot_tree(boom, feature_names = attributes, class_names = classes)
#plt.figure(figsize=(20,20))
plt.savefig("tree.pdf",dpi=2000,format="pdf")



#Split the data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(trainX, trainy, test_size=0.5)
trainerrors = []
testerrors = []
#Compute the train and test errors per depth.
for i in range(2,21):
    treetwo = tree.DecisionTreeClassifier(max_depth=i,criterion='gini')
    treetwo = treetwo.fit(X_train,y_train)
    predictedX_train=treetwo.predict(X_train)
    sum = 0
    for index in range(0,len(predictedX_train)):
        if not (predictedX_train[index]==y_train[index]):
            sum+=1
    trainerrors.append( sum/len(y_train))
    predictedX_test=treetwo.predict(X_test)
    sum = 0
    for index in range(0,len(predictedX_test)):
        if not (predictedX_test[index]==y_test[index]):
            sum+=1
    testerrors.append(sum/len(y_test))
#Plot the train and test errors in one plot.
"""plt.plot(range(2,21),trainerrors,'b-', label = 'Trainingerrors')
plt.xlabel("Depth of the tree")
plt.ylabel("Error")
plt.title("A plot of the training and test classification error")
plt.plot(range(2,21),testerrors,'r-', label = 'Testerrors')
plt.legend()
plt.show()
print("Figure 1: A plot of the training and test classification error of the HTRU2.")"""
