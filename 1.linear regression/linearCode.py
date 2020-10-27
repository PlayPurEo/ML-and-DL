# author : 'wangzhong';
# date: 13/10/2020 15:23

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
print(data.head())
print(data.describe())
print(data.info())

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
plt.show()


# 线性回归的代价函数
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# 让我们在训练集中添加一列，以便我们可以使用向量化的解决方案来计算代价和梯度。
data.insert(0, 'Ones', 1)
print(data.head())

# data.shape为(97,3),tuple数组,shape[1]则是取列的数量
cols = data.shape[1]
# X是所有行，去掉最后一列
X = data.iloc[:, 0:cols - 1]
# X是所有行，最后一列
y = data.iloc[:, cols - 1:cols]
# 检查一下X和y的数据
print(X.head())
print(y.head())
# 将dataframe转换为matrix矩阵
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))
# X为（97，2），97行，97个数据，2列，2个特征值，其中一个为常数,theta转置，2行1列，2行即为系数
error = (X * theta.T) - y
# 尝试计算下代价函数（即为损失函数的平均值）
print(computeCost(X, y, theta))


# 批量梯度下降
def gradientDescent(X, y, theta, alpha, iters):
    # temp为[[0,0]],用来存放迭代后的theta
    temp = np.matrix(np.zeros(theta.shape))
    # ravel将多维降为一维，而这里theta实际上就是（1，2），parameter为2
    parameters = int(theta.ravel().shape[1])
    # ndarray类型，100个0
    cost = np.zeros(iters)

    for i in range(iters):
        # 计算出每一组数据的误差，每行都是样本的误差
        error = (X * theta.T) - y

        for j in range(parameters):
            # 线性回归方程对theta求导后的公式为(theta.T*X-y)*xj的平方的sum
            # 所以这里用X*theta.T，即列向保存，再乘以X中对应的每一列，方便保存求和。
            # multiply为数组对应元素相乘，所以这个term就是还未相加的对theta求导后的数据
            term = np.multiply(error, X[:, j])
            # 求出迭代后的theta
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost


# 初始化学习速率和迭代次数
alpha = 0.01
iters = 1000

g, cost = gradientDescent(X, y, theta, alpha, iters)
# g为迭代完之后的theta
print(g)

# 横坐标
x = np.linspace(data.Population.min(), data.Population.max(), 100)
# 线性回归的最终方程
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

# 代价函数的曲线
# fig代表绘图窗口(Figure)；ax代表这个绘图窗口上的坐标系(axis)，一般会继续对ax进行操作
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

#第二份数据同样的操作，但是因为features的数据值差距过大，要做归一化处理
# path =  'ex1data2.txt'
# data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
# data2.head()
# data2 = (data2 - data2.mean()) / data2.std()
# data2.head()

# 也可以直接引用sklearn模块里的线性回归函数
from sklearn import linear_model
model = linear_model.LinearRegression()
# 模型训练
model.fit(X, y)

# x = np.array(X[:, 1].A1)
x = X[:, 1].A1
# 预测值
fff = model.predict(X)
# (97,1)
print(fff.shape)
f = model.predict(X).flatten()
# (97,)，上面的fff是二维数组，有两个中括号，这个是一维数组，只有一个中括号
print(f.shape)

# 长度为2，即行数。一维数组，len就是数据的量
print(len(np.zeros(100).reshape(2, 50)))

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()