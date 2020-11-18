# author : 'wangzhong';
# date: 08/11/2020 13:03

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


# 对于每个样本，找到所属簇类的索引
def find_label(X, centros):
    # 每个样本所属的k
    idx = []
    for i in range(len(X)):
        # (k,)
        dist = np.linalg.norm((X[i] - centros), axis=1)
        id_i = np.argmin(dist)
        idx.append(id_i)
    return np.array(idx)


def compute_centros(X, idx, k):
    centroids = []
    for i in range(k):
        # 每个簇类的点的均值
        cluster_i = X[idx == i]
        if len(cluster_i) != 0:
            centros_i = np.mean(cluster_i, axis=0)
            centroids.append(centros_i)

    return np.array(centroids)


def run_kmeans(X, centros, iter):
    k = len(centros)
    centros_all = []
    centros_all.append(centros)
    centros_i = centros
    idx = np.array([])
    for i in range(iter):
        idx = find_label(X, centros_i)
        centros_i = compute_centros(X, idx, k)
        centros_all.append(centros_i)

    return idx, np.array(centros_all)


# centros_all为三维数组，第一维为迭代的次数，第二维为k的种类，第三维为特征数
def plot_data(X, idx, centros_all):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=idx, cmap='rainbow')
    plt.plot(centros_all[:,:,0], centros_all[:,:,1],'kx--')


# 随机初始化中心位置和个数
def init_centros(X, k):
    index = np.random.choice(len(X), k)
    return X[index]


if __name__ == '__main__':
    data = loadmat("ex7data2.mat")
    X = data['X']

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    centros = np.array([[3, 3], [6, 2], [8, 5]])
    idx = find_label(X, centros)
    # print(idx)

    # print(compute_centros(X, idx, k=3))

    idx_final, centros_all = run_kmeans(X, centros, iter=10)
    plot_data(X, idx_final, centros_all)
    plt.show()

    # print(init_centros(X, k=3))

    for i in range(4):
        idx, centros_all = run_kmeans(X, init_centros(X, k=3), iter=10)
        plot_data(X, idx, centros_all)
        plt.show()