# author : 'wangzhong';
# date: 01/11/2020 14:43

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# 线性可分svm，主要是看C系数带来的决策边界的变化
data = loadmat("ex6data1.mat")
X, y = data['X'], data['y']


def plot_data():
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='jet')
    plt.xlabel('x1')
    plt.ylabel('y1')


from sklearn.svm import SVC

svc1 = SVC(C=1, kernel='linear')
svc1.fit(X, y.flatten())
# print(svc1)
print(svc1.predict(X))
print(svc1.score(X, y.flatten()))


def plot_boundary(model):
    x_min, x_max = -0.5, 4.5
    y_min, y_max = 1.3, 5
    # 这里xx和yy的shape为(500,500)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    # 一维时，列增加，（250000， 2）
    target = np.c_[xx.flatten(), yy.flatten()]
    z = model.predict(target)
    # (500,500)
    zz = z.reshape(xx.shape)
    # 绘制等高线
    plt.contour(xx, yy, zz)

# 这个等高线暂时不是很理解，这个画法摘自网络
plot_boundary(svc1)
plot_data()
plt.show()

#C = 1/lambda, C越大，则lambda越小，表示正则化很低,容易过拟合
svc100 = SVC(C=100, kernel='linear')
svc100.fit(X, y.flatten())
print(svc100.score(X, y.flatten()))
plot_boundary(svc100)
plot_data()
plt.show()