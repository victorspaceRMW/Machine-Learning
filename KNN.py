# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 04:11:37 2019

@author: wrm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
iris=load_iris()
#print (iris.data)
#print (iris.target)

"""
Attribute Information:\n    
    - sepal length in cm\n    
    - sepal width in cm\n 
    - petal length in cm\n      
    - petal width in cm\n  
"""

x=pd.DataFrame(iris.data)
y=pd.DataFrame(iris.target)
all_data=pd.concat([x,y],axis=1)
"""
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
"""
#print (all_data)
all_data.columns=["sl","sw","pl","pw","label"]
#print (all_data)
#sns.pairplot(all_data,hue="label")
"""
This is about how to use the seaborn-pairplot.
https://yq.aliyun.com/articles/581263
"""

"""
Here, we will start the implementation of KNN
"""
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)

from sklearn.neighbors import KNeighborsClassifier as knc
nbrs=knc(n_neighbors=5)
nbrs.fit(x_train,y_train.values.ravel())
y_pred=nbrs.predict(x_test)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, y_pred))

"""
Here let us draw the confusion_matrix
"""
from sklearn.metrics import confusion_matrix
print (confusion_matrix(y_test,y_pred))
#scores=cvs(nbrs,x,y,cv=5)
#print (scores)
cm=confusion_matrix(y_test,y_pred)

"""
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix_IRIS'); 
#ax.xaxis.set_ticklabels(['business', 'health']); ax.yaxis.set_ticklabels(['health', 'business']);
"""

"""
Here let us calculate the classifcation_report
"""
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.datasets import load_digits
digits=load_digits()
x_digits=digits.data
y_digits=digits.target

x_digits_train,x_digits_test,y_digits_train,y_digits_test=train_test_split(x_digits,y_digits)
nbrs.fit(x_digits_train,y_digits_train)
y_digits_predict=nbrs.predict(x_digits_test)
print (accuracy_score(y_digits_predict,y_digits_test))
cm2=confusion_matrix(y_digits_predict,y_digits_test)
print (cm2)
print (classification_report(y_digits_predict,y_digits_test))


ax2= plt.subplot()
sns.heatmap(cm2, annot=True); #annot=True to annotate cells
# labels, title and ticks
ax2.set_xlabel('Predicted labels');ax2.set_ylabel('True labels'); 
ax2.set_title('Confusion Matrix_Hand_Written_Digits'); 
