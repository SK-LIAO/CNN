# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:31:49 2022

@author: A90127
"""

import numpy as np

#卷積層運算
class Conv:
    # n 層 dim X dim 的 filters
    def __init__(self,dim,n):
        n = int(abs(n))
        #防呆,將dim限制在3,5,7
        ind = np.argmin([abs(dim-i) for i in [3,5,7]])
        dim = [3,5,7][ind]
        #filter的dimension
        self.dim_filters = dim
        #filter個數
        self.num_filters = n
        #建立一個隨機亂數的filters
        self.filters = np.random.randn(n, dim, dim)
    
    #給定圖像矩陣的各個卷積小矩陣 的 生成器   
    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h-self.dim_filters+1):
            for j in range(w-self.dim_filters+1):
                im_region = image[i:(i+self.dim_filters), j:(j+self.dim_filters)]
                yield im_region, i, j
            
    def forward(self,_input):
        self.last_input = _input
        h, w = _input.shape
        output = np.zeros((h-self.dim_filters+1, w-self.dim_filters+1, self.num_filters))
        for im_region, i, j in self.iterate_regions(_input):
            output[i, j] = np.sum(im_region*self.filters, axis=(1, 2))
        return output
    
    def backprop(self,dL_df,lr):
        dL_dx = np.zeros(self.filters.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                dL_dx[f] += dL_df[i, j, f] * im_region
                
        self.filters -= lr * dL_dx
        return None