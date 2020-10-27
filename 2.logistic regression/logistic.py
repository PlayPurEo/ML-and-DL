# author : 'wangzhong';
# date: 17/10/2020 22:22

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 照常，先读取数据观察一下
path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
print(data.head())
# ---------------------------

# 将label为0和1的放在两个dataframe中，由于这里只有0和1，所以这里用==1或者==2也行
# 这里positive和negative并没有重置dataframe的索引，索引还是根据原始的data变量里的
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]
# positive = positive.reset_index(drop=True)
# iloc索引以行号为索引，含头不含尾，loc索引以index或者说标签为索引，含头含尾！
# print(positive.iloc[0:1, :])
# 画个图看一下
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()
# ---------------------------


# 实现sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 实现逻辑回归的代价函数，两个部分，-y(log(hx)和-(1-y)log(1-hx)
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))


# 初始化theta、X、y
data.insert(0, 'Ones', 1)
# 看下data共有多少列
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]
# ndarray， 一维数组
theta = np.zeros(3)

X = np.array(X.values)
y = np.array(y.values)

print(cost(theta, X, y))
# ---------------------------


# 梯度计算函数
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    # 这里的矩阵即使使用ravel，也不是一维数组，因为它是个矩阵，不知道用ravel代表着什么
    # parameters为3
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        # 乘以每个样本中对应每个theta的x
        term = np.multiply(error, X[:, i])
        # 每个位置对应要变更的theta的值
        grad[i] = np.sum(term) / len(X)

    return grad


# 使用scipy内置函数自主确定最佳iter和alpha
import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
print(result)

print(cost(result[0], X, y))

# 最后画一下决策曲线
# 30为起点，100为终点，画100个点
plotting_x1 = np.linspace(30, 100, 100)
plotting_h1 = (- result[0][0] - result[0][1] * plotting_x1) / result[0][2]

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(plotting_x1, plotting_h1, 'y', label='Prediction')
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

# 使用模型来预测
print(sigmoid(result[0].dot([1, 45, 55])))