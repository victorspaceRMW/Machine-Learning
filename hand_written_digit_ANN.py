# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 21:56:56 2019

@author: wrm
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
digits=load_digits()
x=digits.data
y=digits.target

"""
one-hot编码：pd.get_dummies()
That is because there are no sequence for 0~9. All the numbers are equal.
"""
#print (pd.get_dummies(y))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.neural_network import MLPClassifier as MLP

clf=MLP(hidden_layer_sizes=(100,),activation="logistic",solver="adam",
        learning_rate_init=0.0001,max_iter=2000)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, y_pred))
#print (y_test)
#print (y_pred)


"""
We will do experiments on hidden_layer_sizes, activation, solver, 
learning_rate_init and max_iter with parametergrid. However, this is not 
very suitable for the beginners. That is magic! WULa param_grid
"""
"""
from sklearn.model_selection import ParameterGrid as pg
param_grid=pg({"hidden_layer_sizes":[(50,),(100,),(200,)],"learning_rate_init":[0.0001,0.001,0.01]})
#print (list(param_grid))
for params in param_grid:
    clf=MLP(**params)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print (accuracy_score(y_test,y_pred))
"""

"""
Let us test the hidden_layer_sizes first
"""
for term in [(50,),(100,),(200,)]:
    clf=MLP(hidden_layer_sizes=term,activation="logistic",solver="adam",
        learning_rate_init=0.0001,max_iter=2000)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    #from sklearn.metrics import accuracy_score
    print (accuracy_score(y_test, y_pred))
"""
With the similar path, we could test other parameters
"""
    