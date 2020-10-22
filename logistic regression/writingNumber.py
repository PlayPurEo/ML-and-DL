import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

# 使用逻辑回归进行手写数字的识别

data = loadmat("ex3data1.mat")
raw_X = data['X']
raw_y = data['y']
# print(data)
# 看下sample和label的数量，像素为20*20
# print(data['X'].shape)
# print(data['Y'].shape)


# 用向量化去迭代，不用for循环，所以之前写的代码可以完全拿过来用
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 这里的损失函数加上了正则化，但是不对theta0做正则化处理
# 为了使用scipy.optimize，theta必须放在第一个参数
def cost(theta, X, y, lamda):
    # 这里的@就是采用矩阵乘法，且不用把X和theta转为matrix
    # 也就是说，没有必要先转为matrix，再用*去相乘
    # 第三点，采用的是矩阵乘法，类型仍然为array
    A = sigmoid(X@theta)
    # 因为两个都是array，直接用*就能相应位置相乘
    # y和theta的维度必须保持一致！！
    # 如果theta是二维，A@theta也是二维，y就得是二维，才能相应位置相乘
    # 如果theta是一维，A@theta也是一维，y就得是一维，才能相应位置相乘
    # 至于到底是一维还是二维，甚至是logistic.py里那样完全用matrix，都不影响用sum去求和！
    first = -(y*np.log(A))
    second = (1 - y)*np.log(1 - A)
    # reg就是正则化项，lamda/2m,乘以所有theta的平方的和，但是不包括theta0
    reg = (lamda / (2 * len(X))) * np.sum(np.power(theta[1:], 2))
    return np.sum(first - second) / len(X) + reg


# 梯度下降函数，这里不需要alpha和迭代次数，全部交给scipy，只要计算出需要减去的梯度向量即可
# 这里返回的也是一维数组，因minimize要求，且theta0是不参与正则化的
def gradient_reg(theta, X, y, lamda):
    reg = theta[1:] * (lamda/len(X))
    reg = np.insert(reg, 0, values=0, axis=0)
    first = X.T@(sigmoid(X@theta) - y) / len(X)
    return first + reg


X = np.insert(raw_X, 0, values=1, axis=1)
# 因为scipy函数要求，theta变为了一维，那么X@theta也变成了一维，所以这里y也要用一维
y = raw_y.flatten()

def one_vs_all(X,y,lamda,K):
    # 取列，即特征个数
    n = X.shape[1]
    theta_all = np.zeros((K,n))
    # 十个分类器，从1开始到10，10对应数字0
    for i in range(1,K+1):
        theta_i = np.zeros(n)
        result = minimize(fun=cost,
                          x0=theta_i,
                          args=(X, y == i, lamda),
                          method='TNC',
                          jac=gradient_reg)
        theta_all[i-1,:] = result.x

    return theta_all

lamda = 1
K = 10
theta_final = one_vs_all(X,y,lamda,K)

# print(theta_final)

def predict(X, theta_final):
    h = sigmoid(X@theta_final.T)
    h_argmax = np.argmax(h, axis=1)
    # 返回对应的标签，为索引+1
    return h_argmax+1


y_predict = predict(X, theta_final)
acc = np.mean(y_predict == y)
print(acc)