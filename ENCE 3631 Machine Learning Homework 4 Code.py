# -*- coding: utf-8 -*-
"""
Created on Sun May 16 10:59:17 2021

@author: IanSi
"""

import numpy as np
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

boston = load_boston()
X = boston.data
y = boston.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.208,random_state=0)

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)


"""
Least Mean Square: Problem 1 part A
"""

#Training Least square regression,

n,m = X_train.shape
A_train = np.hstack((np.ones((n,1)),X_train))

theta = np.linalg.inv(np.dot(A_train.T,A_train)).dot(A_train.T).dot(y_train)

#Mean-Square Test Error
n_test,m_test = X_test.shape
A_test = np.hstack((np.ones((n_test,1)),X_test))
MSE = (np.linalg.norm(y_test - np.dot(A_test,theta), ord=2)**2)/n_test
print(MSE)



"""
Ridge Regression: Problem 1 part B
"""

#Getting biased by small values of lambda
X_fit,X_holdout,y_fit,y_holdout = train_test_split(X_train,y_train,test_size=0.308,random_state=0)

MSE_min = float('inf')
#Assign a curve, theta, from a particular lambda, and test with holdout set
n,m = X_fit.shape
n_holdout,m_holdout = X_holdout.shape
A_holdout = np.hstack((np.ones((n_holdout,1)),X_holdout))
A_fit = np.hstack((np.ones((n,1)),X_fit))
for q in range(1,31,1):
    lamda = 2**q
    Gamma = ((lamda**0.5)*np.identity(m+1))
    Gamma[0][0] = 0
    theta = np.linalg.inv(np.dot(A_fit.T,A_fit) + (np.dot(Gamma.T,Gamma))).dot(A_fit.T).dot(y_fit)

    MSE = (np.linalg.norm(y_holdout - np.dot(A_holdout,theta), ord=2)**2)/n_holdout
    if(MSE < MSE_min) and (np.linalg.norm(np.dot(A_holdout,theta))<10):
        q_min = q
        MSE_min = MSE
        theta_optimal = theta

#Mean-Square Test Error
n_test,m_test = X_test.shape
A_test = np.hstack((np.ones((n_test,1)),X_test))
MSE = (np.linalg.norm(y_test - np.dot(A_test,theta_optimal), ord=2)**2)/(n_test)
print(MSE)



"""
LASSO: Problem 1 part C
"""

from sklearn import linear_model

X_fit,X_holdout,y_fit,y_holdout = train_test_split(X_train,y_train,test_size=0.308,random_state=0)

lamda = 1
MSE_min = float('inf')
for q in range(-10,21,1):
    reg = linear_model.Lasso(alpha = 2*q)
    reg.fit(X_fit,y_fit)
    B = reg.predict(X_holdout)
    n_holdout = B.size
    MSE = np.linalg.norm(B,ord=2)**2/n_holdout
    if(MSE < MSE_min):
        MSE_min = MSE
        lamda = 2*q
reg = linear_model.Lasso(alpha = lamda)
reg.fit(X_train,y_train)
B = reg.predict(X_test)
n_test = B.size
MSE = np.linalg.norm(B,ord=2)**2/n_test
numZeros = reg.coef_
print(MSE)


"""
Outliers: Problem 2
"""

from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
np.random.seed(2020)
n = 100
xtrain = np.random.rand(n)
ytrain = 0.25 + 0.5*xtrain + np.sqrt(0.1)*np.random.randn(n)
idx = np.random.randint(0,100,10)
ytrain[idx] = ytrain[idx] + np.random.randn(10)


"""
Problem 2 part A
"""

#Construct A matrix
B = np.ones([n,1])
A = np.insert(B, 1, xtrain, axis=1)
lamda = 11
Gamma = (lamda*np.identity(2))
Gamma[1][1] = 0

theta = np.linalg.inv(np.dot(A.T,A)+Gamma).dot(A.T).dot(ytrain)
plt.plot(t1,theta[0] + theta[1]*t1,'r',linewidth=3.0)



"""
Problem 2 part B
"""

reg = linear_model.HuberRegressor(epsilon = 2.50, alpha=0.050)
reg.fit(xtrain.reshape(-1,1),ytrain)

## Plot the training points
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
plt.scatter(xtrain, ytrain, cmap=cmap_light)
plt.xlim(xtrain.min(), xtrain.max())
plt.ylim(ytrain.min(), ytrain.max())
plt.title("training data points")
plt.scatter(xtrain[idx], ytrain[idx], cmap=cmap_bold)
## plot the target function
t1 = np.arange(0.0, 1.0, 0.01)
plt.plot(t1,0.25 + 0.5*t1,'k',linewidth=3.0)

plt.plot(t1,reg.intercept_ + reg.coef_*t1,'r',linewidth=3.0)

plt.legend(['target function','Linear fit','inliers','outliers'])
## Show the plot
plt.show()


