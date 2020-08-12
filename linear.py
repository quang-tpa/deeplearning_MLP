# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 22:42:35 2020

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("data_linear.csv", delimiter=",", dtype=float, skip_header=1)
N = data.shape[0]
X = data[:,0:1]
y = data[:, 1]

W = np.array([1.])
b = np.array([1.])

numIterator = 100
learning_rate = 0.000001 
L = np.ones(numIterator)
for i in range(numIterator):
    y_predict = np.dot(X,W) + b
    r = y_predict - y
    L[i] = np.sum(0.5*r*r)
    W -= learning_rate * np.dot(X.T, r)
    b -= learning_rate * np.sum(r)

plt.scatter(X,y)
y_predict = np.dot(X,W) + b
plt.plot(X, y_predict, 'r')
plt.show()
