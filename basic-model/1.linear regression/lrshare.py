# author : 'wangzhong';
# date: 19/11/2020 16:35

"""
该项目为使用线性回归对股票价格进行预测
数据集来自三方库tushare
"""

import tushare as ts
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# 获取茅台的股票数据
stock = ts.get_hist_data('600519')
# 处理残缺值 1.补0 2.直接扔掉  3.填充
# dropna直接扔掉
stock.dropna(axis=0, inplace=True)

# 按照索引排序，true为升序
stock.sort_index(ascending=True, inplace=True)

# 将high这一列取出，用作label
data_label = pd.DataFrame(stock['high'])
# 列名重命名为label
data_label.columns = ['label']
# 该列向上平移，用作当天预测下一天的label
data_label = data_label.shift(-1)

# 水平合并两个表
stock_final = pd.concat([stock, data_label], axis=1)
# 删除最后一行
stock_final.drop(labels=stock_final.index[len(stock_final) - 1], inplace=True)

stock_data = stock_final.values

# [:,-1]为一维数组，[:,-1:]为二维数组
y = stock_data[:, -1:]
X = np.arange(np.size(y)).reshape(-1, 1)

y_train = y[:-128, :]
X_train = X[:-128, :]
y_test = y[-128:, :]
X_test = X[-128:, :]

model = LinearRegression()
model.fit(X_train, y_train)
y_pre = model.predict(X_test)

plt.plot(X_train, y_train)
plt.plot(X_test, y_test)
# plt.plot(X_test, y_pre)
# 结果极差，R方甚至是负数
# print(model.score(X_test, y_test))

pi_line = Pipeline([('poly', PolynomialFeatures(degree=4)),
                   ('st', StandardScaler()),
                    ('lr', LinearRegression())])
pi_line.fit(X_train, y_train)
# 实时证明，这个根据日期去预测，完全不行，因为有涨有跌
y_pre_poly = pi_line.predict(X_test)
# plt.plot(X_test, y_pre_poly)
# plt.show()

X_multi = stock_data[:, :-1]
X_multi_train = X_multi[:-128, :]
X_multi_test = X_multi[-128:, :]

model.fit(X_multi_train, y_train)
y_multi_pre = model.predict(X_multi_test)
plt.plot(X_test, y_multi_pre)
plt.show()
print(model.score(X_multi_test, y_test))
