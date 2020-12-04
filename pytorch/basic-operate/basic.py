# author : 'wangzhong';
# date: 03/12/2020 20:41

import torch
import numpy as np

# 创建一个未初始化矩阵
empty = torch.empty(5, 3)
# 创建一个空矩阵
zeros = torch.zeros(5, 4)
# 创建指定范围整形矩阵
randInt = torch.randint(1, 10, (5, 4))
# 创建随机矩阵
rand = torch.rand(5, 4)
# 转为tensor类型
x = torch.tensor([5, 5, 3])
# 转为1矩阵
x = x.new_ones((5, 3))
# 生成形状形同，均值为0，方差为1的标准正态分布随机数
x = torch.randn_like(x, dtype=torch.float)
print(x)
# 查看维度
print(x.size())
# 同numpy的reshape
print(x.view(15))
# 从torch转为numpy，内存共享，改变一个，另一个同时改变
a = torch.ones(5, 3)
b = a.numpy()
b[0][0] = 100
print(a)
# 从numpy转为torch，内存共享，改变一个，另一个同时改变
a = np.ones((5, 3))
b = torch.from_numpy(a)
b[0][0] = 100
print(a)
# 查看gpu版是否安装成功
print(torch.cuda.is_available())
