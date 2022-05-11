# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:07:26 2022

@author: A90127
"""

import numpy as np

#軟化層運算
class Softmax:

    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes)
        self.biases = np.random.randn(nodes)

    def forward(self, _input):
        self.last_input_shape = _input.shape
        
        _input = _input.flatten()
        self.last_input = _input
    
        input_len, nodes = self.weights.shape
    
        totals = np.dot(_input, self.weights) + self.biases
        self.last_totals = totals
        t_exp = np.exp(totals)
        return t_exp / np.sum(t_exp)
    
    def backprop(self,lr,label):
        # e^totals
        t_exp = np.exp(self.last_totals)
        # Sum of all e^totals
        S = np.sum(t_exp)
        dL_dt = np.array([t_exp[i]/S-1 if i==label else t_exp[i]/S for i in range(10)])
        
        dt_dw = np.zeros((len(self.last_input),10,10))
        for j in range(len(self.last_input)):
            for i in range(10):
                dt_dw[j,i,i] = self.last_input[j]
        dL_dw = np.dot(dL_dt,dt_dw)
        
        dL_db = dL_dt
        
        dt_dh = self.weights.T
        dL_dh = np.dot(dL_dt,dt_dh)
        
        #更新 w b
        self.weights -= lr * dL_dw
        self.biases -= lr * dL_db
        return dL_dh.reshape(self.last_input_shape)