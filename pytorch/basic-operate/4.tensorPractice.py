# author : 'wangzhong';
# date: 22/12/2020 20:06

"""
练习下常见的tensor格式
"""
from torch import tensor

# 这是一个标量，就是一个值
scalar = tensor(55)
# 正常的数值操作
scalar *= 2
# 0维
print(scalar.dim())
# 可以查看矩阵的维度，如torch.size([2,3,4])代表三维，且每个维度的大小分别是2，3，4
print(scalar.shape)
# -------------------------------------------

v = tensor([1, 2, 3])
# 一维向量
print(v.dim())
# size方法和属性shape一样
print(v.size())
# -------------------------------------------

matrix = tensor([[1, 2, 3], [4, 5, 6]])
# 二维向量
print(matrix.dim())
print(matrix.shape)

# 矩阵乘法
print(matrix.matmul(matrix.T))
# (2,3) * (3,) = (2,)
print(matrix.matmul(tensor([1, 2, 3])))
# (2,) * (2, 3) = (3,)
print(tensor([1, 2]).matmul(matrix))
# 对应位置相乘
print(matrix * matrix)