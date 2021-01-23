# author : 'wangzhong';
# date: 14/01/2021 20:05

import torch
import torch.nn as nn


class alex(nn.Module):
    def __init__(self):
        super(alex, self).__init__()
        # 输入：3*227*227
        # 输出：96*27*27
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # 输入：96*27*27
        # 输出：256*13*13
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )
        # 输入：256*13*13
        # 输出：384*13*13
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU()
        )
        # 输入：384*13*13
        # 输出：384*13*13
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU()
        )
        # 输入：384*13*13
        # 输出：256*6*6
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )  # out_put_size = 6*6*256 = 9126

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(9126, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 100)  # 几分类就输出几
        )

    # 正向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        outputs = self.dense(x)
        return outputs
