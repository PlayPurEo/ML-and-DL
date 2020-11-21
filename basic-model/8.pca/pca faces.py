# author : 'wangzhong';
# date: 21/11/2020 22:33

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat('ex7faces.mat')
X = data['X']


def plot_100images(X):
    fig, axis = plt.subplots(ncols=10, nrows=10, figsize=(10, 10))
    for c in range(10):
        for r in range(10):
            # 这里注意要转置
            axis[c, r].imshow(X[10*c + r].reshape(32, 32).T)
            axis[c, r].set_xticks([])
            axis[c, r].set_yticks([])
    plt.show()


# 画出来瞅一眼
# plot_100images(X)
X_mean = np.mean(X, axis=0)
X_demean = X - X_mean
# 协方差矩阵
C = X_demean.T @ X_demean / len(X_demean)
U, S, V = np.linalg.svd(C)

U1 = U[:, :36]
X_reduction = X_demean@U1
X_restore = X_reduction@U1.T + X_mean

plot_100images(X_restore)