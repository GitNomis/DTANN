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
import treeprint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

classes = ["RFI","Pulsar"]
attributes = ["Mean of the integrated profile","Standard deviation of the integrated profile","Excess kurtosis of the integrated profile","Skewness of the integrated profile","Mean of the DM-SNR curve","Standard deviation of the DM-SNR curve","Excess kurtosis of the DM-SNR curve","Skewness of the DM-SNR curve"]

def hyperparameter(trainX, trainy):  
    averageTrainErrors = []
    averageTestErrors = []
    #Create a splitter that splits the shuffled data in 10 partitions.
    splitter = KFold(n_splits = 10, shuffle = True)
    #Split the data using the splitter and compute the errors per fold.
    for train, test in splitter.split(trainX, trainy):
        trainerrors = []
        testerrors = []
        X_train, X_test, y_train, y_test = trainX[train], trainX[test], trainy[train], trainy[test]
        #Compute the train and test errors per depth.
        for i in range(2,21):
            treetwo = tree.DecisionTreeClassifier(max_depth=i,criterion='gini')
            treetwo = treetwo.fit(X_train,y_train)
            predictedX_train=treetwo.predict(X_train)
            sum = 0
            for index in range(0,len(predictedX_train)):
                if not (predictedX_train[index]==y_train[index]):
                    sum+=1
            trainerrors.append(sum/len(y_train))
            predictedX_test=treetwo.predict(X_test)
            sum = 0
            for index in range(0,len(predictedX_test)):
                if not (predictedX_test[index]==y_test[index]):
                    sum+=1
            testerrors.append(sum/len(y_test))
        averageTrainErrors.append(trainerrors)
        averageTestErrors.append(testerrors)
    averageDepthTestErrors = []
    averageDepthTrainErrors = []
    #Compute the average train and test errors over 10 folds per depth.
    for i in range(0,19):
        sumtest = 0
        sumtrain = 0
        for errors in averageTrainErrors:
            sumtrain += errors[i]
        averageDepthTrainErrors.append(sumtrain/len(averageTrainErrors))
        for errors in averageTestErrors:
            sumtest += errors[i]    
        averageDepthTestErrors.append(sumtest/len(averageTestErrors))  
    #Plot the errors in one plot.        
    plt.plot(range(2,21),averageDepthTrainErrors,'b-', label = 'Trainingerrors')
    plt.xlabel("Depth of the tree")
    plt.ylabel("Average error")
    plt.title("A plot of the average training and test classification error using 10-fold")
    plt.plot(range(2,21),averageDepthTestErrors,'r-', label = 'Testerrors')
    #plt.legend()
    plt.show()
   
data = pd.read_excel('.\HTRU_2.xlsx',header=None)
X = data.iloc[:,:8].values
y = data.iloc[:,8:].values.ravel()
negX=np.array([X[i] for i in range (0,len(X)) if y[i]==0])
posX=np.array([X[i] for i in range (0,len(X)) if y[i]==1])

trainX= np.concatenate((posX,negX[sample_without_replacement(len(negX),len(posX))]),axis=0)
trainy= np.array([1 for x in posX]+[0 for x in posX] )  

boom = tree.DecisionTreeClassifier(criterion='gini',min_samples_split=100, max_depth=3)
boom = boom.fit(trainX, trainy)
#treeprint.tree_print(boom, attributes, classes)
#tree.plot_tree(boom, feature_names = attributes, class_names = classes)

falsepos = 0
truepos = 0
falseneg = 0
trueneg = 0

splitter = KFold(n_splits = 10, shuffle = True)
for train, test in splitter.split(trainX, trainy):
    X_train, X_test, y_train, y_test = trainX[train], trainX[test], trainy[train], trainy[test]
    hyperparameter(X_train, y_train)
    treetwo = tree.DecisionTreeClassifier(max_depth=4,criterion='gini')
    treetwo = treetwo.fit(X_train,y_train)
    predictedX_test=treetwo.predict(X_test)
    for index in range(0,len(predictedX_test)):
        if predictedX_test[index]== 0 and y_test[index]==0:
            trueneg +=1
        if predictedX_test[index]== 1 and y_test[index]==0:
            falsepos+=1
        if predictedX_test[index]== 1 and y_test[index]==1:
            truepos+=1
        if predictedX_test[index]== 0 and y_test[index]==1:
            falseneg +=1
print("falsepos ", falsepos/10, "truepos ", truepos/10, "falseneg ", falseneg/10, "trueneg ", trueneg/10)
        
    
