# author : 'wangzhong';
# date: 27/10/2020 19:39

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt

data = loadmat("ex5data1.mat")
# 训练集（12，1）
X_train, y_train = data['X'], data['y']
# 验证集 （21，1）
X_val, y_val = data['Xval'], data['yval']
# 测试集 (21,1
X_test, y_test = data['Xtest'], data['ytest']
# 加入截距项
X_train = np.insert(X_train, 0, 1, axis=1)
X_val = np.insert(X_val, 0, 1, axis=1)
X_test = np.insert(X_test, 0, 1, axis=1)


# 简单看下数据情况
def plot_data():
    fig, ax = plt.subplots()
    ax.scatter(X_train[:, 1], y_train)
    ax.set(xlabel="water level",
           ylabel="water flowing out")


# plot_data()

def reg_cost(theta, X, y, lamda):
    cost = np.sum(np.power((X @ theta - y.flatten()), 2))
    reg = np.sum(np.power(theta[1:], 2)) * lamda
    return (cost + reg) / (2 * len(X))


# theta = np.ones(X_train.shape[1])
# lamda = 1
# print(reg_cost(theta, X_train, y_train, lamda))

def reg_gradient(theta, X, y, lamda):
    grad = (X @ theta - y.flatten()) @ X
    reg = lamda * theta
    reg[0] = 0
    return (grad + reg) / (len(X))


# print(reg_gradient(theta, X_train, y_train, lamda))


def train_model(X, y, lamda):
    theta = np.ones(X.shape[1])
    res = minimize(fun=reg_cost,
                   x0=theta,
                   args=(X, y, lamda),
                   method='TNC',
                   jac=reg_gradient)
    return res.x


def plot_learning_curve(X_train, y_train, X_val, y_val, lamda):
    x = range(1, len(X_train) + 1)
    train_cost = []
    cv_cost = []
    for i in x:
        res = train_model(X_train[:i, :], y_train[:i, :], lamda)
        train_cost_i = reg_cost(res, X_train[:i, :], y_train[:i, :], lamda)
        cv_cost_i = reg_cost(res, X_val, y_val, lamda)
        train_cost.append(train_cost_i)
        cv_cost.append(cv_cost_i)

    plt.plot(x, train_cost, label='training cost')
    plt.plot(x, cv_cost, label='cv cost')
    plt.legend()
    plt.xlabel('training number')
    plt.ylabel('error')
    plt.show()


# 不考虑正则化的情况，出现欠拟合的情况
plot_learning_curve(X_train, y_train, X_val, y_val, lamda=0)


# 增加多项式，从x的平方到x的多次方
def poly_feature(X, power):
    for i in range(2, power + 1):
        X = np.insert(X, X.shape[1], np.power(X[:, 1], i), axis=1)
    return X


def get_standard(X):
    # 按行计算，即求每一列的均值和方差
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    return means, stds


def feature_normalize(X, means, stds):
    X[:, 1:] = (X[:, 1:] - means[1:]) / stds[1:]
    return X


# 测试，最大六次方
power = 6
X_train_poly = poly_feature(X_train, power)
X_val_poly = poly_feature(X_val, power)
X_test_poly = poly_feature(X_test, power)

train_means, train_stds = get_standard(X_train_poly)
X_train_norm = feature_normalize(X_train_poly, train_means, train_stds)
X_val_norm = feature_normalize(X_val_poly, train_means, train_stds)
X_test_norm = feature_normalize(X_test_poly, train_means, train_stds)

theta_fit = train_model(X_train_norm, y_train, lamda=0)


def plot_poly_fit():
    plot_data()
    x = np.linspace(-60, 60, 100)
    xReshape = x.reshape(100, 1)
    xReshape = np.insert(xReshape, 0, 1, axis=1)
    xReshape = poly_feature(xReshape, power)
    xReshape = feature_normalize(xReshape, train_means, train_stds)

    plt.plot(x, xReshape @ theta_fit, 'r--')
    plt.show()


plot_poly_fit()

# 高方差，次数太多，过拟合
plot_learning_curve(X_train_norm, y_train, X_val_norm, y_val, lamda=0)
# 加入正则化
plot_learning_curve(X_train_norm, y_train, X_val_norm, y_val, lamda=1)
# 正则化很大，欠拟合
plot_learning_curve(X_train_norm, y_train, X_val_norm, y_val, lamda=100)

lamdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost = []
cv_cost = []
for lamda in lamdas:
    res = train_model(X_train_norm, y_train, lamda)
    tc = reg_cost(res, X_train_norm, y_train, lamda=0)
    cv = reg_cost(res, X_val_norm, y_val, lamda=0)

    training_cost.append(tc)
    cv_cost.append(cv)

plt.plot(lamdas, training_cost, label='training cost')
plt.plot(lamdas, cv_cost, label='cv cost')
plt.legend()
plt.show()

#拿到最佳lamda
bestLamda = lamdas[np.argmin(cv_cost)]
#用最佳lamda来训练测试集
res = train_model(X_train_norm, y_train, bestLamda)
print(reg_cost(res, X_test_norm, y_test, lamda=0))
