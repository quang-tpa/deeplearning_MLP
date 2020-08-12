# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 00:04:19 2020

@author: Admin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
learning_rate = 0.9

class ActivationLayer(object):
    def __init__(self, name, activation_type):
        self.name = name
        self.activation_type = activation_type
        supported_activations = ['sigmoid','relu','leaky_relu','softmax','linear']
        if self.activation_type not in supported_activations:
            raise ValueError('Activation Function {} not supported'.format(self.activation_function))
        
    def activation_function(self, input_val):
        if self.activation_type=='relu':
            return np.maximum(0,input_val)
        elif self.activation_type=='leaky_relu':
            return  np.where(input_val > 0, input_val, input_val * 0.01)
        elif self.activation_type=='linear':
            return input_val
        # clipping input to avoid thrown nans on overflow  underflow
        input_val = np.clip(input_val, -500, 500 )
        if self.activation_type=='sigmoid':
            return 1 / (1+np.exp(-input_val))
        elif self.activation_type=='softmax':
            exp_input_val = np.exp(input_val)
            return exp_input_val / np.sum(exp_input_val, axis = 1, keepdims=True)
    
    def gradient_activation_function(self, input_val):
        output_val = np.copy(input_val)
        # just get the derivative of each function and evaluate
        if self.activation_type=='relu':
            output_val[input_val>0] = 1
            output_val[input_val<0] = 0
            output_val[input_val==0] = 0.5  
            return output_val
        elif self.activation_type=='leaky_relu':
            output_val[input_val>0] = 1
            output_val[input_val<0] = 0.01
            output_val[input_val==0] = 0.5  
            return output_val
        elif self.activation_type=='linear':
            return 1
        elif self.activation_type=='sigmoid':
            return self.activation_function(output_val) * (1 - self.activation_function(output_val))
        elif self.activation_type=='softmax':
            return 1
    def forward(self, input_val):
        return self.activation_function(input_val)
    
    def backward(self,prev_pre_activation_gradients, previous_weights):
        return  np.dot(previous_weights.T , prev_pre_activation_gradients)
    

# Lớp MLP
class MLP:
    def __init__(self, layers, learning_rate=0.1):
        # Mô hình layer ví dụ [2,2,1]
      self.layers = layers 
      
      # Hệ số learning_rate
      self.learning_rate = learning_rate
        
      # Tham số W, b
      self.W = []
      self.b = []
      
      # Giá trị của các activate function
      self.A = []

      # Khởi tạo các tham số ở mỗi layer
      for i in range(0, len(layers) - 1):
            self.W.append(np.ones((layers[i], layers[i+1])))
            self.b.append(np.zeros((1, layers[i+1])))
            
    
    # Tóm tắt mô hình neural network
    def summary(self):
        return "Neural network [{}]".format("-".join(str(l) for l in self.layers))
    
    
    # Train mô hình với dữ liệu
    def fit_partial(self, X, y):
        #Activate function container
        
        
        # quá trình feedforward
        self.A = self.predict(X)
        
        dA = [self.gradient_output_layer(y)]
        
        # quá trình backpropagation and Gradient descent
        for i in reversed(range(0, len(self.layers)-1)):
            dA.append(self.backward(dA[-1], i))
        
    
    def fit(self, X, y, epochs=20, verbose=10):
        for epoch in range(0, epochs):
            self.fit_partial(X, y)
            if epoch % verbose == 0:
                loss = self.calculate_loss(X, y)
                sel = np.argmax(self.A[-1], axis=1)
                accuracy = np.mean(sel == y)
                print("Epoch {}, loss {}, accuracy {}".format(epoch, loss, accuracy))
    
    # Dự đoán
    def predict(self, X):
        A = [X]
        for i in range(len(self.layers) - 2):
            A.append(self.forward(A[-1], i, "relu"))
        A.append(self.forward(A[-1], -1, "softmax"))
        return A

    # Tính loss function
    def calculate_loss(self, X, y):
        y_predict = self.A[-1]
        return self.loss_function(y_predict, y)/y.shape[0]
        
    def cross_entropy(seft, y, y_hat):
        return -np.sum(y * np.log(y_hat))
    
    def forward(self, x, ilayer, typeLayer):
        a = ActivationLayer("ok", typeLayer)
        return a.forward(np.dot(x, self.W[ilayer]) + self.b[ilayer])
    
    def backward(self, da, ilayer, typeLayer = "relu"):
        dW = np.dot(self.A[ilayer].T, da)
        db = np.sum(da, axis=0, keepdims=True)
        dz = np.dot(da, self.W[ilayer].T)
        dz[self.A[ilayer] <= 0] = 0
        
        self.W[ilayer] -= learning_rate*dW
        self.b[ilayer] -= learning_rate*db
        return dz
    
    def gradient_output_layer(self, y):
        da = self.A[-1].copy()
        da[range(y.shape[0]), y] -= 1
        return da/y.shape[0]
    
    def loss_function(self, y_predict, y):
        return np.sum(self.cross_entropy(1, y_predict[range(y.shape[0]), y]))
    
# data = pd.read_csv("iris_full.csv").values
# X = data[:, 2:4]
# num_Example = X.shape[0]
# X = preprocessing.scale(X)
# y = data[:, 4].astype(int)

# #Tạo MLP vơi mô hình 2, 50, 3
# p = MLP([X.shape[1], 50, 3], 0.9)
# p.fit(X, y, 100, 1)


# plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
# plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
# plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], color='g', label='2')
# plt.legend()

# h = 0.02
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], p.W[0]) + p.b[0]), p.W[1]) + p.b[1]
# Z = np.argmax(Z, axis=1)
# Z = Z.reshape(xx.shape)
# fig = plt.figure()
# plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.xlim(xx.min(), xx.max())

    
# Mượn tạm tensorflow 
import tensorflow as tf

# load data
(x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# normalize
x = x / 255.0
n = len(x)
X = np.array([x[i].flatten() for i in range(n)])

#Tạo MLP vơi mô hình 2, 2, 2,3
p = MLP([784 ,10], 0.1)
p.fit(X, y, 100, 1)


    
    
    
    