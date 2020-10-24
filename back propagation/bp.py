# author : 'wangzhong';
# date: 23/10/2020 22:16

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt

data = loadmat("ex4data1.mat")
raw_X = data['X']
raw_y = data['y']
X = np.insert(raw_X, 0, values=1, axis=1)


# 对y进行one-hot处理，把y转换为长度为10的向量
# 5000 * 10
def one_hot_encode(raw_y):
    result = []
    for i in raw_y:
        y_template = np.zeros(10)
        y_template[i - 1] = 1
        result.append(y_template)

    return np.array(result)


theta = loadmat("ex4weights.mat")
theta1 = theta['Theta1']
theta2 = theta['Theta2']


# 序列化theta，全部变成一维数组，放在一起
def serialize(a, b):
    return np.append(a.flatten(), b.flatten())


# 解序列化
def deserialize(theta):
    theta1 = theta[:25*401].reshape(25,401)
    theta2 = theta[25*401:].reshape(10,26)
    return theta1, theta2


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 前向传播
def feed_forward(theta, X):
    theta1, theta2 = deserialize(theta)
    a1 = X
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, values=1, axis=1)
    z3 = a2 @ theta2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h


# 不带正则化的损失函数
def cost(theta, X, y):
    a1, z2, a2, z3, h = feed_forward(theta, X)
    J = -np.sum(y*np.log(h) + (1-y)*np.log(1-h))/len(X)
    return J


# d带正则化的损失函数
def reg_cost(theta, X, y, lamda):
    sum1 = np.sum(np.power(theta1[:,1:], 2))
    sum2 = np.sum(np.power(theta2[:,1:], 2))
    reg = lamda/(2 * len(X)) * (sum1 + sum2)
    return reg + cost(theta, X, y)


# sigmoid函数求导：sigmoid(z)' = sigmoid(z) * (1-sigmoid(z))
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


# 无正则化梯度下降
def gradient(theta, X, y):
    theta1, theta2 = deserialize(theta)
    a1, z2, a2, z3, h = feed_forward(theta, X)
    d3 = h - y
    d2 = d3 @ theta2[:,1:] * sigmoid_gradient(z2)
    # (5000,10)   (5000,26)
    D2 = (d3.T @ a2) / len(X)
    D1 = (d2.T @ a1) / len(X)
    return serialize(D1, D2)


# 正则化梯度下降
def reg_gradient(theta, X, y, lamda):
    theta1, theta2 = deserialize(theta)
    reg1 = lamda / len(X) * theta1[:, 1:]
    reg2 = lamda / len(X) * theta2[:, 1:]
    D = gradient(theta, X, y)
    D1, D2 = deserialize(D)
    D1[:, 1:] = D1[:, 1:] + reg1
    D2[:, 1:] = D2[:, 1:] + reg2
    return serialize(D1, D2)


# 用scipy进行优化
# 不加正则化，正确率为99.9%，显然过拟合
def nn_training(X, y):
    # 10285为所有的theta的数量
    init_theta = np.random.uniform(-0.5, 0.5, 10285)
    res = minimize(fun=cost,
                   x0=init_theta,
                   args=(X,y),
                   method='TNC',
                   jac=gradient,
                   options={'maxiter':300})
    return res


y = one_hot_encode(raw_y)
# res = nn_training(X, y)
raw_y = data['y'].reshape(5000,)

# _,_,_,_,h = feed_forward(res.x, X)
# y_predict = np.argmax(h, axis=1) + 1
# acc = np.mean(y_predict == raw_y)
# print(acc)


# 加入正则化，正确率降低，一定程度上解决了过拟合
def reg_nn_training(X, y, lamda):
    init_theta = np.random.uniform(-0.5, 0.5, 10285)
    res = minimize(fun=reg_cost,
                   x0=init_theta,
                   args=(X, y, lamda),
                   method='TNC',
                   jac=reg_gradient,
                   options={'maxiter': 300})
    return res


lamda = 10
res = reg_nn_training(X, y, lamda)
_,_,_,_,h = feed_forward(res.x, X)
y_predict = np.argmax(h, axis=1) + 1
acc = np.mean(y_predict == raw_y)
print(acc)


# 显示隐藏层特征向量
def plot_hidden_layer(theta):
    theta1, theta2 = deserialize(theta)
    hidden_layer = theta1[:,1:] # 25*400，不需要偏置项
    fig,ax = plt.subplots(nrows=5, ncols=5, figsize=(8, 8), sharex=True, sharey=True)
    for i in range(5):
        for j in range(5):
            ax[i, j].imshow(hidden_layer[5*i+j].reshape(20, 20).T, cmap='gray_r')

    plt.xticks([])
    plt.yticks([])
    plt.show()


plot_hidden_layer(res.x)