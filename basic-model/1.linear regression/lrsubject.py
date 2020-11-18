# author : 'wangzhong';
# date: 17/11/2020 12:53

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('owid-covid-data.csv')
data_france = data[data['location'] == 'France']
france = data_france[:]['total_cases'].values.astype(int)
X = np.arange(france.size).reshape(-1, 1)

poly_feature = Pipeline(steps=[('poly', PolynomialFeatures(degree=20)),  # 特征扩展 degree是几次方
                               ('St_sc', StandardScaler()),  # 归一化 (x-均值)/方差
                               ('lr', LinearRegression())])

poly_feature.fit(X, france)
y = poly_feature.predict(X)
print(poly_feature.score(X, france))
plt.plot(X, y, color='r')

model = LinearRegression()
model.fit(X, france)
w = model.coef_
theta = model.intercept_
y1 = w * X + theta
print('score:' + str(model.score(X, france)))

plt.scatter(X, france)
plt.plot(X, y1)
plt.show()
