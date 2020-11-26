# author : 'wangzhong';
# date: 27/11/2020 00:40

"""
吴恩达推荐系统算法
Nm = 电影数量
Nu = 用户的数量
y(i,j)表示用户j对电影i的评价
r(i,j)表示用户j是否对电影i做出了评价，1表示做出了评价，0表示没有
通过协同过滤同步更新特征值和theta
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def serialize(X, Theta):
    """
    序列化参数，scipy方法要求
    :param X:
    :param Theta:
    :return:
    """
    return np.append(X.flatten(), Theta.flatten())


def deserialize(params, nu, nm, nf):
    """
    解序列化函数
    :param params:
    :param nu:
    :param nm:
    :param nf:
    :return:
    """
    X = params[:nm * nf].reshape(nm, nf)
    Theta = params[nm * nf:].reshape(nu, nf)
    return X, Theta


def costFunction(params, Y, R, nu, nm, nf, lamda):
    """
    损失函数
    :param params:
    :param Y:
    :param R:
    :param nu:
    :param nm:
    :param nf:
    :param lamda:
    :return:
    """
    X, Theta = deserialize(params, nu, nm, nf)
    # 这里要点乘R，也就是对应位置相乘，R里为0的说明没打分，是需要预测的，没有误差一说
    error = 0.5 * np.square((X @ Theta.T - Y) * R).sum()
    reg1 = 0.5 * lamda * np.square(X).sum()
    reg2 = 0.5 * lamda * np.square(Theta).sum()
    return error + reg1 + reg2


def costGradient(params, Y, R, nu, nm, nf, lamda):
    """
    梯度函数
    :param params:
    :param Y:
    :param R:
    :param nu:
    :param nm:
    :param nf:
    :param lamda:
    :return:
    """
    X, Theta = deserialize(params, nu, nm, nf)
    X_gradient = ((X @ Theta.T - Y) * R) @ Theta + lamda * X
    theta_gradient = ((X @ Theta.T - Y) * R).T @ X + lamda * Theta
    return serialize(X_gradient, theta_gradient)


def normalizeRatings(Y, R):
    """
    均值化处理
    :param Y:
    :param R:
    :return:
    """
    # 这里求均值后是一维数组，为了方便，reshape成二维
    Y_mean = (Y.sum(axis=1) / R.sum(axis=1)).reshape(-1, 1)
    Y_norm = (Y - Y_mean) * R
    return Y_norm, Y_mean


if __name__ == '__main__':
    data = loadmat("ex8_movies.mat")
    Y, R = data['Y'], data['R']

    param = loadmat("ex8_movieParams.mat")
    # nu为用户数量，为943
    # nm为电影数量，为1682
    # nf为电影特征数量，为10
    # theta为用户特征的权重矩阵，为（943，10）
    # X为电影特征矩阵为（1682，10）
    X, Theta, nu, nm, nf = param['X'], \
                           param['Theta'], \
                           param['num_users'], \
                           param['num_movies'], \
                           param['num_features']
    nu = int(nu)
    nm = int(nm)
    nf = int(nf)
    # 用小范围的用户和电影先看下costFunction
    users = 4
    movies = 5
    features = 3
    X_sub = X[:movies, :features]
    Y_sub = Y[:movies, :users]
    theta_sub = Theta[:users, :features]
    R_sub = R[:movies, :users]
    cost_test = costFunction(serialize(X_sub, theta_sub), Y_sub, R_sub, users, movies, features, 0)

    # 创建一个新用户，初始化对电影的评分
    my_ratings = np.zeros((nm, 1))
    my_ratings[9] = 5
    my_ratings[66] = 5
    my_ratings[96] = 5
    my_ratings[121] = 4
    my_ratings[148] = 4
    my_ratings[285] = 3
    my_ratings[490] = 4
    my_ratings[599] = 4
    my_ratings[643] = 4
    my_ratings[958] = 5
    my_ratings[1117] = 3
    Y = np.c_[Y, my_ratings]
    R = np.c_[R, my_ratings != 0]
    nm, nu = Y.shape

    # 对Y进行均值归一化
    Y_norm, Y_mean = normalizeRatings(Y, R)
    # 初始化参数
    X_ran = np.random.random((nm, nf))
    theta_ran = np.random.random((nu, nf))
    param_ran = serialize(X_ran, theta_ran)
    lamda = 10
    res = minimize(fun=costFunction,
                   x0=param_ran,
                   args=(Y_norm, R, nu, nm, nf, lamda),
                   method="TNC",
                   jac=costGradient,
                   options={'maxiter': 100})
    param_fit = res.x
    # 训练得到的X特征值和theta权重
    X_fit, theta_fit = deserialize(param_fit, nu, nm, nf)

    Y_predict = X_fit @ theta_fit.T
    y_pre = Y_predict[:, -1] + Y_mean.flatten()
    # 这里是降序排行，所以加个[::-1]
    index = np.argsort(y_pre)[::-1]
    with open('movie_ids.txt', 'r', encoding='gb18030') as f:
        movies = []
        for line in f:
            tokens = line.strip().split(" ")
            movies.append(" ".join(tokens[1:]))
    for i in range(10):
        print(index[i], movies[index[i]], y_pre[index[i]])
