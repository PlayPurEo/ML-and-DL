# author : 'wangzhong';
# date: 01/11/2020 16:34

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 确定C和gamma的最佳值
data = loadmat('ex6data3.mat')
X, y = data['X'], data['y']
Xval, yval = data['Xval'], data['yval']



def plot_data():
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='jet')
    plt.xlabel('x1')
    plt.ylabel('y1')


def plot_boundary(model):
    x_min, x_max = -0.6, 0.4
    y_min, y_max = -0.7, 0.8
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


Cvalues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gammas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

best_score = 0
best_params = (0, 0)
# 这里还可以把每组的值都打出来
for c in Cvalues:
    for gamma in gammas:
        svc = SVC(C=c, kernel='rbf', gamma=gamma)
        svc.fit(X, y.flatten())
        score = svc.score(Xval, yval.flatten())
        if score > best_score:
            best_score = score
            best_params = (c, gamma)
print(best_score, best_params)

svc2 = SVC(C=best_params[0], kernel='rbf', gamma=best_params[1])
svc2.fit(X, y.flatten())
plot_boundary(svc2)
plot_data()
plt.show()