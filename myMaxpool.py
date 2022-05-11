# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:18:28 2022

@author: A90127
"""

import numpy as np


#池化層(此取最大，但也可取平均)運算
class MaxPool: 
    #給定影像的各個池化小矩陣 的 生成器  
    def iterate_regions(self, image):
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2
        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j
    
    def forward(self, _input):
        self.last_input = _input
        h, w, num_filters = _input.shape
        output = np.zeros((h // 2, w // 2, num_filters))
        for im_region, i, j in self.iterate_regions(_input):
          output[i, j] = np.amax(im_region, axis=(0, 1))
        return output
    
    def backprop(self,dL_dh,lr):
        dL_df = np.zeros(self.last_input.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))
            for i2 in range(h):
                for j2 in range(w):
                    for k2 in range(f):
                        if im_region[i2,j2,k2] == amax[k2]:
                            dL_df[i*2+i2,j*2+j2,k2] = dL_dh[i,j,k2]
                            
        return dL_df