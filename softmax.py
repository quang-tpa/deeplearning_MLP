# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 00:04:19 2020

@author: Admin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

def cross_entropy(y, y_hat):

    # One hot encode Y to create a distribution
    return -np.sum(y * np.log(y_hat))

data = pd.read_csv("data_softmax_iris_full.csv").values
X = data[:, 2:4]
num_Example = X.shape[0]
X = preprocessing.scale(X)
y = data[:, 4].astype(int)
n_input = 2
n_output = 3

W = np.random.randn(n_input, n_output)
b = np.random.randn(n_output,1)

learning_rate = 0.05
num_epochs = 100 
L = np.zeros(num_epochs)


x1 = np.dot(X,W)
x2 = x1 + b

# def forward_softmax(z):

#     exp_z = np.exp(z)
#     a = exp_z / np.sum(exp_z, axis = 1, keepdims=True)

#     return a

# for i in range(num_epochs):
#     y_predict = forward_softmax(np.dot(X,W) + b)
    
#     L[i] = np.sum(cross_entropy(y, y_predict[range(num_Example), y]))
        
#     gradient = y_predict
#     gradient[range(num_Example), y] -= 1

#     W -= learning_rate * np.dot(X.T, gradient)
#     b -= learning_rate*np.sum(gradient, axis=0, keepdims=True)
    
# plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
# plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
# plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], color='g', label='2')

# h = 0.01
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
# Z = np.argmax(Z, axis=1)
# Z = Z.reshape(xx.shape)
# fig = plt.figure()
# plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())