#!/usr/bin/env python
# coding: utf-8

# # Simple Multi-linear Regression with Python

# Here we will first read the original data from the dataset. As there are dataset of the price of houses in Boston in sklearn, we will not import the Boston house price dataset.  
# We will take the big X as the feature and y as the label.

# In[1]:


import numpy as np
from sklearn.datasets import load_boston

X=load_boston().data
y=load_boston().target

print (X.shape)
print (y.shape)


# We will divide the original dataset as two parts, "train" and "test."

# In[2]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)

print (X_train.shape)
print (y_test.shape)


# Now we will import the LinearRegression() from the sklearn.linear_model. In this part we will take the Normalize in LinearRegression as "True." It could not change to the result but it could accelerate the program.

# In[4]:


from sklearn.linear_model import LinearRegression
model1=LinearRegression()
model1.fit(X_train,y_train)

train_score=model1.score(X_train,y_train)
print (train_score)
test_score=model1.score(X_test,y_test)
print (test_score)


# Now we will use the polynomial fitting to fit the model. The only thing you need to adjust is current_degree. That means the degree of the polynomial function that you use to fit the model. 
# For more details you could refer to:
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

# In[19]:


from sklearn.preprocessing import PolynomialFeatures as PF
def poly_fit(degree=1,normalize=True):
    poly=PF(degree,include_bias=False)
    X_train_new=poly.fit_transform(X_train)
    X_test_new=poly.fit_transform(X_test)
    return X_train_new,X_test_new

model2=LinearRegression()
current_degree=2
X_train_new,X_test_new=poly_fit(current_degree)
model2.fit(X_train_new,y_train)
train_score_new=model2.score(X_train_new,y_train)
print (current_degree)
print (train_score_new)
test_score_new=model2.score(X_test_new,y_test)
print (test_score_new)


# In[20]:


print (model2)


# Now let us start to draw the learning curve

# In[13]:


from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

