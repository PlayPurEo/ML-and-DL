# author : 'wangzhong';
# date: 21/11/2020 22:06

"""
sklearn pca实战
"""
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = loadmat('ex7data1.mat')
X = data['X']
X_mean = np.mean(X, axis=0)
X_demean = X - X_mean

# n_components可以为整数，表示需要降的维数K；可以为小数，表示对方差占比的要求
pca = PCA(n_components=0.80)
pca.fit(X_demean)
# 方差占比array
print(pca.explained_variance_ratio_)
# 特征值
print(pca.explained_variance_)
# 特征向量
print(pca.components_)
# 降维后的样本
X_reduction = pca.fit_transform(X_demean)
# print(pca.fit_transform(X))
# 降维后
X_restore = pca.inverse_transform(X_reduction) + X_mean
print(X_restore)
# 测试集的均值归一和特征向量都要用训练集的
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(X_restore[:, 0], X_restore[:, 1])
plt.show()
