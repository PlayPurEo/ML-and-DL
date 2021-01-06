# author : 'wangzhong';
# date: 09/12/2020 02:41

import torch

# 简单的torch自动求导尝试，这里的t = x + b
x = torch.randn(3, 4, requires_grad=True)
# print(x)
b = torch.randn(3, 4, requires_grad=True)
# print(b)
t = x + b
# print(t)
y = t.sum()
# print(y)
y.backward()
# print(x.grad)

x = torch.rand(1)
b = torch.rand(1, requires_grad=True)
w = torch.rand(1, requires_grad=True)
y = w * x
z = y + b
# 判断是否是leaf节点，显然这个最简单的公式里，y不是
print(y.is_leaf)
# retain_graph=True：保留z-y-x的计算图
z.backward(retain_graph=True)
print(w.grad)