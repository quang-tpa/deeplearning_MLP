# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:31:03 2020

@author: Admin
"""

import numpy as np
import os
# library for training progress bar
from tqdm import tqdm_notebook as tqdm
# library used to print model summary 
from prettytable import PrettyTable
from math import log, floor
import matplotlib.pyplot as plt
import pickle

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
            return exp_input_val / np.sum(exp_input_val, axis = 0)
    
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
    def forward(self, input):
        return self.activation_function(input)
    
    def backward(self,prev_pre_activation_gradients, previous_weights):
        return  np.dot(previous_weights.T , prev_pre_activation_gradients)
    

class PreActivationLayer(object):
    def __init__(self,name,dimensions,initialization_type='glorot'):
        self.name = name
        # is the dimension of the weights as tuple ( current layer size , input to the activation fn weight )
        self.dimensions = dimensions 
        self.weights = None
        self.bias = None 
        self.dw = None
        self.db = None
        self.d_preactivation = None
        # we have 3 initialization types zeros, uniform , glorot
        self.initialize_weigths(initialization_type)

    def initialize_weigths(self,initialization_type):
        self.bias = np.zeros((self.dimensions[0],1))
        if initialization_type == 'zeros':
            self.weights = np.zeros(self.dimensions)
        elif initialization_type =='normal':
            self.weights = np.random.normal(0,1,self.dimensions)
        elif initialization_type =='glorot':
            uniform_range = np.sqrt(6) / np.sqrt(np.sum(self.dimensions))
            self.weights = np.random.uniform(-1 * uniform_range ,uniform_range ,self.dimensions)
        else:
            raise ValueError('Not supported Initialization type {}'.format(initialization_type))

    def forward(self,input):
        return np.dot(self.weights, input) + self.bias

    def backward(self, prev_activation_output, prev_weight, gradient_activation, gradient_pre_activation):
        n_items_batch = prev_activation_output.shape[1]
        self.d_preactivation = np.multiply(gradient_activation, gradient_pre_activation)
        self.dw = (1.0/n_items_batch )* np.dot(self.d_preactivation, prev_activation_output.T)
        self.db = (1.0/n_items_batch )* np.sum(self.d_preactivation,axis=1, keepdims=True)

class NeuralLayer(object):
    # its dimension is hidden layer size * its input size
    def __init__(self, name, dimension, parent, activation_type, initialization_type):
        self.name = name
        self.dimension = dimension
        self.pre_activation = PreActivationLayer(name+'_pre_activation',self.dimension,initialization_type)
        self.output_activation=ActivationLayer(name+'_activation',activation_type)

        self.cached_pre_activation_output_forward = None
        self.cached_activation_output_forward = None 

        self.cached_preactivation_gradient = None
        self.cached_activation_gradient = None
    
    def forward(self,input):
        self.cached_pre_activation_output_forward = self.pre_activation.forward(input)
        self.cached_activation_output_forward = self.output_activation.forward(self.cached_pre_activation_output_forward)
        return self.cached_activation_output_forward

    def backward(self,prev_activation_output, prev_weight, prev_pre_activation_gradients):
        gradient_pre_activation = self.output_activation.gradient_activation_function(self.cached_pre_activation_output_forward)

        self.cached_activation_gradient = self.output_activation.backward(prev_pre_activation_gradients, prev_weight)
        
        self.pre_activation.backward(prev_activation_output,prev_weight,self.cached_activation_gradient, gradient_pre_activation)

        self.cached_preactivation_gradient = self.pre_activation.d_preactivation
    
