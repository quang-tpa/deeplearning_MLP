# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 23:17:24 2020

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def cross_entropy(y, y_hat):

    # One hot encode Y to create a distribution
    return -np.sum(y * np.log(y_hat))

def sign_mod(x):
    return 1/(1+np.exp(-x))

data = pd.read_csv("logictic.csv").values

X = data[:, 0:2]
y = data[:, 2]

W = np.ones(X.shape[1])
b = np.array([1.])

learning_rate = 0.0001 
num_epochs = 10000
L = np.ones(num_epochs)
acc = np.ones(num_epochs)

for i in range(num_epochs):
    y_predict = sign_mod(np.dot(X,W) + b)
    L[i] = cross_entropy(y, y_predict) + cross_entropy(1 - y, 1 - y_predict)
    acc[i] = (y_predict == y).mean()
    r = y_predict - y
    W -= learning_rate * np.dot(X.T, r)
    b -= learning_rate * np.sum(r)

plt.scatter(X[:10, 0], X[:10, 1], c='red', edgecolors='none', s=30, label='cho vay')
plt.scatter(X[10:, 0], X[10:, 1], c='blue', edgecolors='none', s=30, label='từ chối')
plt.legend(loc=1)
plt.xlabel('Mức lương (triệu)')
plt.ylabel('Kinh nghiệm (năm)')
predict = np.dot(X, W) + b
plt.plot((4,10), (predict[4], predict[8]))
plt.show()
