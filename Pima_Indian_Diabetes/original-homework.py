# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 20:58:40 2019

@author: wrm
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

"""
The Pima Indian dataset is downloaded from Kaggle:
https://www.kaggle.com/kumargh/pimaindiansdiabetescsv

And here is the describtion towards the problem. The only difference is that I used Python rather than R to do the homework.
This homework is designed by world-famous computer vision expert Prof.David Forysth.

Problem 1
I strongly advise you use the R language for this homework (but word is out on Piazza that you could use Python; note I don't know if packages are available in Python). You will have a place to upload your code with the submission.

The UC Irvine machine learning data repository hosts a famous collection of data on whether a patient has diabetes (the Pima Indians dataset), originally owned by the National Institute of Diabetes and Digestive and Kidney Diseases and donated by Vincent Sigillito. You can find this data at http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes. You should look over the site and check the description of the data. In the "Data Folder" directory, the primary file you need is named "pima-indians-diabetes.data". This data has a set of attributes of patients, and a categorical variable telling whether the patient is diabetic or not. For several attributes in this data set, a value of 0 may indicate a missing value of the variable.

Part 1A Build a simple naive Bayes classifier to classify this data set. We will use 20% of the data for evaluation and the other 80% for training. There are a total of 768 data-points.
You should use a normal distribution to model each of the class-conditional distributions. You should write this classifier yourself (it's quite straight-forward).

Report the accuracy of the classifier on the 20% evaluation data, where accuracy is the number of correct predictions as a fraction of total predictions.

Part 1B Now adjust your code so that, for attribute 3 (Diastolic blood pressure), attribute 4 (Triceps skin fold thickness), attribute 6 (Body mass index), and attribute 8 (Age), it regards a value of 0 as a missing value when estimating the class-conditional distributions, and the posterior. R uses a special number NA to flag a missing value. Most functions handle this number in special, but sensible, ways; but you'll need to do a bit of looking at manuals to check.
Report the accuracy of the classifier on the 20% that was held out for evaluation.

Part 1C Now use the caret and klaR packages to build a naive bayes classifier for this data, assuming that no attribute has a missing value. The caret package does cross-validation (look at train) and can be used to hold out data. You should do 10-fold cross-validation. You may find the following fragment helpful
train (features, labels, classifier, trControl=trainControl(method='cv',number=10))

The klaR package can estimate class-conditional densities using a density estimation procedure that I will describe much later in the course. I have not been able to persuade the combination of caret and klaR to handle missing values the way I'd like them to, but that may be ignorance (look at the na.action argument).
Report the accuracy of the classifier on the held out 20%

Part 1-D Now install SVMLight, which you can find at http://svmlight.joachims.org, via the interface in klaR (look for svmlight in the manual) to train and evaluate an SVM to classify this data. For training the model, use:
svmlight (features, labels, pathsvm)

You don't need to understand much about SVM's to do this as we'll do that in following exercises. You should NOT substitute NA values for zeros for attributes 3, 4, 6, and 8.
Using the predict function in R, report the accuracy of the classifier on the held out 20%

Hint If you are having trouble invoking svmlight from within R Studio, make sure your svmlight executable directory is added to your system path. Here are some instructions about editing your system path on various operating systems: https://www.java.com/en/download/help/path.xml You would need to restart R Studio (or possibly restart your computer) afterwards for the change to take effect.

"""

"""
Part1A
"""
df=pd.read_csv("C:/Users/wrm/Desktop/Pima Indian Diabetes/pima-indians-diabetes.csv",header=None)
#print (df)
x=df.drop([8],axis=1)
y=df[8]
ynew=pd.get_dummies(y)
#print (ynew)
#print (x)
#print (y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#print (X_train)
#print (y_test)

from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(X_train,y_train)
y_predict=clf.predict(X_test)

from sklearn.metrics import accuracy_score
print ("Part1A:",accuracy_score(y_predict,y_test))

"""
Part1B
"""
df[2].replace(0,np.NaN,inplace=True)
df[3].replace(0,np.NaN,inplace=True)
df[5].replace(0,np.NaN,inplace=True)
"""
It is very clumsy to assign 0 to the real attribute.
"""
#print (df)

"""
Method1. Drop the row directly. You could find that, only 70% of the data are remained.
"""
df_method1=df.dropna()
#print (df_method1)
"""
Method2. Replace the NaN with average
"""
df_method2=df.fillna(df.mean())
x=df_method2.drop([8],axis=1)
y=df_method2[8]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
y_predict=clf.predict(X_test)
print ("Part1B:",accuracy_score(y_predict,y_test))

"""
From the answer you could clearly notice that there is an obvious increase of the
accuracy score.
"""

"""
Part1D
"""
from sklearn.svm import SVC
clf_svr=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf_svr.fit(X_train,y_train)
y_predict=clf_svr.predict(X_test)
#print (y_test)
#print (y_predict)
print ("Part1D:",accuracy_score(y_predict,y_test))
"""
For this one, it just predict every attribute to 0. So it is not a practical 
algorithm here.
"""
