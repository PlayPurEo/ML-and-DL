# author : 'wangzhong';
# date: 22/11/2020 22:42

"""
吴恩达异常检测作业
用F1-score作为评判标准
"""
from typing import Tuple

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


# 获取均值和方差
# 这里指定入参类型和返回类型
def meanAndVar(X: np.ndarray, isVariance: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param X: 二维数组
    :param isVariance: 是否为协方差
    :return: 均值和方差
    """
    means = np.mean(X, axis=0)
    if isVariance:
        sigma2 = (X - means).T @ (X - means) / len(X)
    else:
        sigma2 = np.var(X, axis=0)
    return means, sigma2


def gaussion(X, means, sigma2):
    """
    :param X: 二维数组
    :param means: 均值
    :param sigma2: 方差，可能是协方差矩阵，需要做判断
    :return: 多元正态密度函数
    """
    # 如果不是协方差矩阵，变为对角线矩阵
    if np.ndim(sigma2) == 1:
        sigma2 = np.diag(sigma2)
    X = X - means
    n = X.shape[1]

    # 多元正太密度函数前半部分和后半部分
    first = np.power(2*np.pi, (-n/2)) * (np.linalg.det(sigma2) ** (-0.5))
    second = np.diag(X @ np.linalg.inv(sigma2) @ X.T)
    p = first * np.exp(-0.5*second)
    p = p.reshape(-1, 1)
    return p


# 画等高线
def plot_gaussion(X, means, sigma2):
    # 长度为60的一维数组
    x = np.arange(0, 30, 0.5)
    y = np.arange(0, 30, 0.5)
    # (60, 60)二维数组
    xx, yy = np.meshgrid(x, y)
    z = gaussion(np.c_[xx.ravel(), yy.ravel()], means, sigma2)
    zz = z.reshape(xx.shape)
    plt.plot(X[:, 0], X[:, 1], 'bx')
    contour_levels = [10**h for h in range(-20, 0, 3)]
    plt.contour(xx, yy, zz, contour_levels)


# 选取最佳阈值
def selectThreshold(yval, p):
    bestEpsilon = 0
    bestF1 = 0
    epsilons = np.linspace(min(p), max(p), 1000)
    for i in epsilons:
        # True == 1, False == 0
        p_ = p < i
        tp = np.sum((yval == 1) & (p_ == 1))
        fp = np.sum((yval == 0) & (p_ == 1))
        fn = np.sum((yval == 1) & (p_ == 0))
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        F1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = i
    return bestEpsilon, bestF1


if __name__ == '__main__':
    data = loadmat('ex8data1.mat')
    X = data['X']
    X_val, y_val = data['Xval'], data['yval']

    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()

    # means, sigma2 = meanAndVar(X, False)
    # plot_gaussion(X, means, sigma2)
    # means, sigma2 = meanAndVar(X, True)
    # plot_gaussion(X, means, sigma2)

    means, sigma2 = meanAndVar(X, True)
    p_val = gaussion(X_val, means, sigma2)
    bestEpsilon, bestF1 = selectThreshold(y_val, p_val)
    p = gaussion(X, means, sigma2)
    # 这里是generator表达式
    anoms = np.array([X[i] for i in range(len(X)) if p[i] < bestEpsilon])
    # 看一下这个阈值下，检测的异常点
    plot_gaussion(X, means, sigma2)
    plt.scatter(anoms[:, 0], anoms[:, 1])
    plt.show()