# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 11:09:38 2020

@author: pille

Doc: 
    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

Cite (APA):
    Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), 2825-2830.    

"""
from sklearn.neural_network import MLPClassifier
import pandas as pd

data = pd.read_excel('..\HTRU_2.xlsx',header=None)
X = data.iloc[:,:8].values
y = data.iloc[:,8:].values.ravel()

NN = MLPClassifier()
NN.fit(X,y)
sum = 0
for i in y:
    if i == 0:
        sum+=1
print(sum/len(y))        
print(NN.score(X,y))