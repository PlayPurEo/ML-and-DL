# author : 'wangzhong';
# date: 21/01/2021 14:47

"""
损失函数模块
"""

import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)