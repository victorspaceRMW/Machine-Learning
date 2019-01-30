# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 03:47:40 2019

@author: wrm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
"""
This is the start of part1
"""

"""
This part let us import the iris dataset
"""
from sklearn.datasets import load_iris
iris=load_iris()
x=pd.DataFrame(iris.data)
y=pd.DataFrame(iris.target)
iris_all=pd.concat([x,y],axis=1)
iris_all.columns=["sl","sw","pl","pw","label"]
#print (iris_all)
#print (x)
#print (y)
"""
Make a box-plot to show distributions
"""
fig,axs=plt.subplots(figsize=(10,6))
sns.set(style="white", font_scale=1.5)
sns.boxplot(data=iris_all, orient="v", palette="Set2", ax=axs)

from sklearn.preprocessing import scale
x_new=pd.DataFrame(scale(pd.DataFrame(iris.data)))
y_new=y
iris_all_new=pd.concat([x_new,y],axis=1)
iris_all_new.columns=["sl","sw","pl","pw","label"]
#print (iris_all_new)

"""
Make a box-plot to show distributions after the rescaling
"""
fig,axs=plt.subplots(figsize=(10,6))
sns.set(style="white", font_scale=1.5)
sns.boxplot(data=iris_all_new, orient="v", palette="Set2", ax=axs)

"""
This is the end of part1
"""

"""
This is the start of part2
"""

#from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
from sklearn import svm
clf=svm.SVC(kernel='linear', C=0.1)
clf.fit(x_train,y_train.values.ravel())
y_pred=clf.predict(x_test)
from sklearn.metrics import classification_report
print (classification_report(y_pred,y_test))

#print (x_train)
#print (scale(x_train))

clf2=svm.SVC(kernel="linear", C=0.1)
clf2.fit(pd.DataFrame(scale(x_train)),y_train.values.ravel())
y_pred_new=clf.predict(pd.DataFrame(scale(x_test)))
print (classification_report(y_pred,y_test))

"""
We could conclude the scale does not effect the final results of SVM.
But I once discussed with senior engineer of Tencent and he once mentioned that
SVM could effect the result? What is the right situation here?
This is the end of part2
"""

"""
This is the start of part3.
Feature selection.
We use two of the feature selection algorithm.
https://scikit-learn.org/stable/modules/feature_selection.html#tree-based-feature-selection
1. RFE
2. RFC
"""
from sklearn.feature_selection import RFE
svc=svm.SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=3, step=1)
rfe.fit(x_train,y_train.values.ravel())
print (rfe.ranking_)
"""
This could give a result as [1 2 1 1] which means the second feature is the 
least important
"""
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train.values.ravel())
print (rfc.feature_importances_)
"""
[0.06113438 0.04994848 0.52502338 0.36389376]
So we could conclude that feature 1, 2 are the least important feature.
"""
