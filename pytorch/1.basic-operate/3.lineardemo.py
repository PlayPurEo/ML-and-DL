# author : 'wangzhong';
# date: 13/12/2020 13:56

"""
基于pytorch的线性回归demo练习
"""
import numpy as np
import torch
import torch.nn as nn


# 继承nn.module，实现前向传播，线性回归即是全连接层
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


if __name__ == '__main__':
    # torch里要求必须是float
    x = np.arange(1, 12, dtype=np.float32).reshape(-1, 1)
    y = 2 * x + 1

    model = LinearRegressionModel(1, 1)
    # 可以通过to()或者cuda()使用GPU进行模型的训练，需要将模型和数据都转换到GPU上，也可以指定具体的GPU，如.cuda(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    epochs = 1000
    learning_rate = 0.01
    # 优化迭代
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    # 评估指标，均方误差
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        inputs = torch.from_numpy(x).to(device)
        labels = torch.from_numpy(y).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # 进行权重更新
        optimizer.step()

        if epoch % 50 == 0:
            print("epoch:", epoch, "loss:", loss.item())

    predict = model(torch.from_numpy(x).requires_grad_()).data.numpy()
    # torch.save(model.state_dict(), "model.pk1")
    result = model.load_state_dict(torch.load("model.pk1"))