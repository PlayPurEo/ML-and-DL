# author : 'wangzhong';
# date: 21/01/2021 14:49

"""
模型评估函数
"""

import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        # item是返回tensor里的数值，脱离了tensor类型，纯数值
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


if __name__ == '__main__':
    a = torch.tensor([1, 0], dtype=torch.float, requires_grad=True)
    b = torch.tensor([1, 0], dtype=torch.float, requires_grad=True)
    c = torch.sum(a == b)
    print(c)
    print(c.data)
    print(c.item())
    d = 5
    print((d + c.item()))