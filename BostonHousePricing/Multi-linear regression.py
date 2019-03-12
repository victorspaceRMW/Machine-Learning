#!/usr/bin/env python
# coding: utf-8

# # Simple Multi-linear Regression with Python

# Here we will first read the original data from the dataset. As there are dataset of the price of houses in Boston in sklearn, we will not import the Boston house price dataset.  
# We will take the big X as the feature and y as the label.

# In[7]:


import numpy as np
from sklearn.datasets import load_boston

X=load_boston().data
y=load_boston().target

print (X.shape)
print (y.shape)


# We will divide the original dataset as two parts, "train" and "test."

# In[8]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)

print (X_train.shape)
print (y_test.shape)


# Now we will import the LinearRegression() from the sklearn.linear_model. In this part we will take the Normalize in LinearRegression as "True." It could not change to the result but it could accelerate the program.

# In[10]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

train_score=model.score(X_train,y_train)
print (train_score)
test_score=model.score(X_test,y_test)
print (test_score)

