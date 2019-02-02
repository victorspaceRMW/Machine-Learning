# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 01:28:24 2019

@author: wrm
"""

import numpy as np
import pandas as pd

df=pd.read_csv("C:/Users/wrm/Desktop/KaggleProject1/train.csv")
df.drop(["Ticket","Embarked","Name","PassengerId"],axis=1,inplace=True)
#print (df)
"""
Titanic travels from:
C->Q->S
Let us delete those travlers who are traving from:
C->Q
C->S
Q->S
It seems that there are no information of those who get out of the shape on S or Q
"""

"""
Here I want to prove that Fare and Pclass are co-linear
"""
#print ((df.loc[df['Pclass'] == 1])["Fare"].describe())
#print ((df.loc[df['Pclass'] == 2])["Fare"].describe())
#print ((df.loc[df['Pclass'] == 3])["Fare"].describe())
"""
Here you could output the fare:
count    216.000000
mean      84.154687
std       78.380373
min        0.000000
25%       30.923950
50%       60.287500
75%       93.500000
max      512.329200
Name: Fare, dtype: float64
count    184.000000
mean      20.662183
std       13.417399
min        0.000000
25%       13.000000
50%       14.250000
75%       26.000000
max       73.500000
Name: Fare, dtype: float64
count    491.000000
mean      13.675550
std       11.778142
min        0.000000
25%        7.750000
50%        8.050000
75%       15.500000
max       69.550000
Name: Fare, dtype: float64
Obviously, these two are co-linear. Maybe the Fare contains other prices that 
would not effect the final result. So we will delete that feature as well.
"""
df.drop(["Fare"],axis=1,inplace=True)
#print (df.columns)
#print (df)
"""
在这花的时间长了：如何把一个dataframe中的多个值替换为一个值？
"""
df.drop(["Cabin"],axis=1,inplace=True)
df["Age"].fillna(df["Age"].mean(),inplace=True)
#print (df)
"""
Let us try replace the Age with the average first
"""
df["Sex"].replace(["male","female"],[0,1],inplace=True)
#print (df)
y=pd.get_dummies(df["Survived"])
x=df.drop(columns=["Survived"],inplace=False)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
#print (x)

"""
Let us try RandomForestClassifier First
"""
from sklearn.ensemble import RandomForestClassifier as RFC
rfc=RFC()
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test,y_pred))
"""
This method could return an accuracy of 0.8
"""