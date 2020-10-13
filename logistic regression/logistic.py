import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 照常，先读取数据观察一下
path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
print(data.head())
# ---------------------------
# 将label为0和1的放在两个dataframe中，由于这里只有0和1，所以这里用==1或者==2也行
# 这里positive和negative并没有重置dataframe的索引，索引还是根据原始的data变量里的
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]
positive = positive.reset_index(drop=True)
# iloc索引以行号为索引，含头不含尾，loc索引以index或者说标签为索引，含头含尾！
# print(positive.iloc[0:1, :])
# 画个图看一下
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()
# ---------------------------

