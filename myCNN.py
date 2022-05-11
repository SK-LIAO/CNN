# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:01:58 2022

@author: A90127
"""

import mnist
import numpy as np
from myConv import Conv
from myMaxpool import MaxPool
from mySoftmax import Softmax


class CNN:
    def __init__(self):
        self.conv = Conv(5,8)                   # 28x28x1 -> 24x24x8
        self.pool = MaxPool()                   # 24x24x8 -> 12x12x8
        self.softmax = Softmax(12 * 12 * 8, 10) # 13x13x8 -> 10
    
    #計算 每個數字的機率out 和差熵損失 loss 和 預測是否準確 acc
    def forward(self,image,label):
        out = self.conv.forward((image / 255) - 0.5)
        out = self.pool.forward(out)
        out = self.softmax.forward(out)
    
        loss = -np.log(out[label])
        acc = 1 if np.argmax(out) == label else 0

        return out, loss, acc
    
    #訓練模型的係數
    def train(self,im,label,lr=0.005):
        # forward 計算
        out, loss, acc = self.forward(im, label)
        #使用隨機梯度下降訓練模型的各係數
        gradient = self.softmax.backprop(lr, label)
        
        gradient = self.pool.backprop(gradient,lr)
        gradient = self.conv.backprop(gradient,lr)
        #回傳 Loss 與 預測是否正確
        return loss, acc
    
    
    
if __name__ == '__main__':
    
    train_images = mnist.train_images()[:7000]
    train_labels = mnist.train_labels()[:7000]
    test_images = mnist.test_images()[7000:]
    test_labels = mnist.test_labels()[7000:]
        
    model = CNN()
    print('MNIST CNN initialized!')

    for epoch in range(30):
        print('--- Epoch %d ---' % (epoch + 1))
        #亂數訓練資料組碼
        permutation = np.random.permutation(len(train_images))
        #隨機挑500個出來訓練
        train_images = train_images[permutation[:500]]
        train_labels = train_labels[permutation[:500]]
        
        #開始訓練
        loss = 0
        num_correct = 0
        for i, (im, label) in enumerate(zip(train_images, train_labels)):
            #每訓練一百次、秀一次結果，並將結果規0重新計算。
            if i > 0 and i % 100 == 99:
                print('[Step {}] Past 100 steps: Average Loss {:.3f} | Accuracy: {}%'.format(i + 1, loss / 100, num_correct))
                loss = 0
                num_correct = 0
            #訓練    
            #_, l, acc = model.forward(im, label)
            l, acc = model.train(im, label)
            loss += l
            num_correct += acc
    
    
    #拿測試組數據測試模型準確度
    print('\n--- Testing the CNN ---')
    loss = 0
    num_correct = 0
    for im, label in zip(test_images, test_labels):
        _, l, acc = model.forward(im, label)
        loss += l
        num_correct += acc
    
    num_tests = len(test_images)
    print('Test Average Loss:', loss / num_tests)
    print('Test Accuracy: {:.2f}%'.format(100*num_correct / num_tests))
    



