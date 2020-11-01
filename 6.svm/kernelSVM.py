# author : 'wangzhong';
# date: 01/11/2020 15:42

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.svm import SVC

data = loadmat('ex6data2.mat')
X, y = data['X'], data['y']


def plot_data():
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='jet')
    plt.xlabel('x1')
    plt.ylabel('y1')


# gamma即为高斯核函数的超参数，为1/σ^2
# 它越大，即σ越小，正态分布越陡，越容易过拟合
svc1 = SVC(C=1, kernel='rbf', gamma=1)
svc1.fit(X, y.flatten())
print(svc1.score(X, y.flatten()))


def plot_boundary(model):
    x_min, x_max = 0, 1
    y_min, y_max = 0.4, 1
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

plot_boundary(svc1)
plot_data()
plt.show()


svc2 = SVC(C=1, kernel='rbf', gamma=100)
svc2.fit(X, y.flatten())
plot_boundary(svc2)
plot_data()
plt.show()