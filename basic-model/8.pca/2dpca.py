# author : 'wangzhong';
# date: 21/11/2020 18:24

"""
吴恩达pca作业
"""
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat('ex7data1.mat')
X = data['X']

# 去均值化
X_demean = X - np.mean(X, axis=0)
# 协方差矩阵
C = X_demean.T@X_demean / len(X_demean)

# U为特征向量， 这里U和V相等
U, S, V = np.linalg.svd(C)
U1 = U[:, 0]
# 实现降维
X_reduction = X_demean@U1

# 矩阵还原
X_restore = X_reduction.reshape(50, 1)@U1.reshape(1,2) + np.mean(X, axis=0)

plt.scatter(X[:, 0], X[:, 1])
plt.scatter(X_restore[:, 0], X_restore[:, 1])
plt.show()

# PCA的效果评估，可以直接通过S特征值矩阵
# print(S[0] / (S[0] + S[1]))