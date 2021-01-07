# author : 'wangzhong';
# date: 06/01/2021 19:31

"""
1.用pytorch搭建一个完整的卷积神经网络
进行mnist手写数字实战
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class numberCNN(nn.Module):
    def __init__(self):
        super(numberCNN, self).__init__()
        # 这是一个卷积网络，包括卷积层，激活函数，池化层
        # 可以把这些放到sequential里，也可以一步一步放在forward里
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(  # 下一个的输入 (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # 输出 (32, 14, 14)
            nn.ReLU(),  # relu层
            nn.MaxPool2d(2),  # 输出 (32, 7, 7)
        )
        # 直接做一层全连接
        self.out = nn.Linear(32 * 7 * 7, 10)  # 全连接层得到的结果

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def accuracy(predictions, labels):
    # predictions的data为一个64*10的矩阵,dim为1，即为按行取最大值
    # 返回值为2个，第0个为最大值tensor，第1个为对应的索引tensor
    # 这里的索引刚好对应了数字本身，如索引是9，对应的数字就是9
    pred = torch.max(predictions.data, 1)[1]
    # view_as，形状变为相同，这里的rights依然是一个tensor
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


def start():
    # 定义超参数
    input_size = 28  # 图像的总尺寸28*28
    num_classes = 10  # 标签的种类数
    num_epochs = 3  # 训练的总循环周期
    batch_size = 64  # 一个batch的大小，64张图片

    # 训练集
    train_dataset = datasets.MNIST(root='./data',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)
    # 测试集，都会自动下载
    test_dataset = datasets.MNIST(root='./data',
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)
    # 构建batch数据

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=True)

    myModel = numberCNN()

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(myModel.parameters(), lr=0.001)

    for epoch in range(num_epochs):

        train_right = []
        for idx, data in enumerate(train_loader):
            inputs, labels = data
            myModel.train()
            outputs = myModel(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            right = accuracy(outputs, labels)
            # 每个批次，将正确个数记录，和总个数
            train_right.append(right)

            if idx % 100 == 0:
                myModel.eval()
                val_right = []
                for valData in test_loader:
                    valInputs, valLabels = valData
                    output = myModel(valInputs)
                    right = accuracy(output, valLabels)
                    val_right.append(right)

                # 准确率计算
                train_r = (sum([tup[0] for tup in train_right]), sum([tup[1] for tup in train_right]))
                val_r = (sum([tup[0] for tup in val_right]), sum([tup[1] for tup in val_right]))

                print("当前epochL", epoch, "当前训练量：", (idx+1)*batch_size)
                print("训练正确个数：", train_r[0].numpy(), "训练总个数：", train_r[1],
                      "训练准确率：", train_r[0].numpy() / train_r[1])
                print("测试正确个数：", val_r[0].numpy(), "测试总个数：", val_r[1],
                      "测试准确率：", val_r[0].numpy() / val_r[1])


if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    output = torch.randn(1, 5, requires_grad=True)
    label = torch.empty(1, dtype=torch.long).random_(5)
