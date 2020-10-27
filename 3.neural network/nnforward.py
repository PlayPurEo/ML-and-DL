# author : 'wangzhong';
# date: 18/10/2020 23:14

import numpy as np
from scipy.io import loadmat

data = loadmat("../2.logistic regression/ex3data1.mat")
raw_X = data['X']
raw_y = data['y']
X = np.insert(raw_X, 0, values=1, axis=1)
y = raw_y.flatten()

theta = loadmat("ex3weights.mat")
# 可用theta.keys()查看有哪些键
# （25，401），（10，26）
theta1 = theta['Theta1']
theta2 = theta['Theta2']


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# input layer
a1 = X

z2 = a1@theta1.T
a2 = sigmoid(z2)
# 在hidden layer也要做一个偏置项
a2 = np.insert(a2, 0, values=1, axis=1)
z3 = a2@theta2.T
a3 = sigmoid(z3)

y_predict = np.argmax(a3, axis=1)
y_predict = y_predict + 1
acc = np.mean(y_predict == y)
print(acc)