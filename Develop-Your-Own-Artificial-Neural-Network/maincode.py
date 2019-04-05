# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 08:40:48 2019

@author: wrm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork:  
    # 通过初始化的方式实现 输入‘x’及输出‘y’、以及网络权重（定义了两层，以及"jiedian"个节点的网络结构，你可以把节点选择为1,2,3,4,....） 的基本结构
    # 如果BP神经网络的层数太多，会导致结果不收敛。因此我们建议选择神经网络中只安排1层或者2层节点
    def __init__(self, x, y,jiedian):
        self.input = x
        # x作为输入，被定义为神经网络中的self.input
        self.y=y
        # y作为输入，被定义为神经网络中的self.y
        self.weights1 = np.random.rand(self.input.shape[1],jiedian) 
        # 首先构造一个大小为（self.input.shape[1],jiedian）的神经网络结构。神经网络的权重都是随机的。
        # 这样的结构，意味着weights1的行数是输入变量x的列数，那么相乘时可能是与常规的 W·X 的方式相反，即 X·W
        # 因此，X的每一行，代表着一次x输入，对应着每一个输出y
        
        self.weights2 = np.random.rand(jiedian,1)
        # 首先构造一个大小为（jiedian，1）的神经网络结构。神经网络的权重都是随机的。
        # 同上的，这样的结构形式定义，代表的表达式或应是：[(X·W1+b1)·W2]+b2
        ## 而在经过上述W1与W2的相乘后，每一行的x对应着一个y_i的值
        self.output = np.zeros(y.shape)

    
    def feedforward(self): #定义前传方法：
        # 注意忽略了每层的偏差 b_i
        # 且转换函数使用 sigmoid()
        # 因没有相应的函数定义，将其中的 sigmoid() 函数进行展开
        # sigmoid(m)=1/(1 + np.exp(-m))
       
        npDotW1_Inpt = np.dot(self.input, self.weights1)
        self.layer1 = 1/(1+np.exp(-1*npDotW1_Inpt))
        
        npDotL1_W2 = np.dot(self.layer1, self.weights2)
        self.output = 1/(1+np.exp(-1*npDotL1_W2))
        
    def backprop(self): # 定义后传方法：
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        # 以上是原文注解，即在向后传播 backprop 时，使用链式法则获得相应权重值的损失函数
        # 因没有相应的函数定义，将原贴中的 sigmoid_derivative() 函数进行展开
        # sigmoid_derivative(m)=m * (1 - m)

        ## 此处采用的是 s_deriv(x) = x*(1-x) 的方式，如采用 exp(-x)/(1+exp(-x))^2) 的解析式也可以，本质相同：
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * (self.output*(1-self.output))))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * (self.output*(1-self.output)), self.weights2.T) * (self.layer1*(1-self.layer1))))
    
        # update the weights with the derivative (slope) of the loss function
        # 通过上面的求导斜率，对W1和W2进行更新：
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        
    def return_matrix(self):
        return ("weights1:",self.weights1),("weights2:",self.weights2)
    
        
# 建立输入和输出变量：
x_input = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y_obj = np.array([[0,1,1,0]]).T

# 实例化 ANN：
nn = NeuralNetwork(x_input, y_obj,jiedian=3)

# 进行计算，迭代次数为1500次：
m = 1500
loss = np.zeros(m)

for i in range(m):
    nn.feedforward() # 前传计算结果
    nn.backprop() # 后传更新权重

    loss[i] = np.sum((nn.output-y_obj)**2) # 记录每次的结果偏差

print ("Here is the matrix")
print (nn.return_matrix())
print ("Here is the reuslt")
print (nn.output)

# 绘制结果图形：
plt.plot(loss)
plt.xlabel('Iteration')
plt.ylabel('LossValue')
plt.grid(True)
