# author : 'wangzhong';
# date: 18/11/2020 13:13

"""
该项目为利用sklearn自带boston房价数据进行线性回归预测实战
"""
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

boston = load_boston()
X = boston.get('data')
y = boston.get('target')
features = boston.get('feature_names')

# for i in range(len(features)):
#     plt.scatter(X[:,i], y)
#     plt.title(features[i])
#     plt.show()

# 做一个标准化
st = StandardScaler()
st.fit_transform(X)

# 直接进行切分，test数据占比为20%，即4：1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
y_pre = model.predict(X_test)

print(model.score(X_test, y_test))
