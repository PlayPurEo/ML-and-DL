# author : 'wangzhong';
# date: 25/01/2021 20:17
import os
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt
from dlmodel.vgg.tools.common_tools import get_vgg16
from dlmodel.vgg.tools.dataset import CatDogDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
log_dir = os.path.join("..", "results", time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


if __name__ == '__main__':
    # config
    data_dir = os.path.join("..", "..", "DATA", "train")
    path_state_dict = os.path.join("..", "data", "vgg16-397923af.pth")
    num_classes = 2

    MAX_EPOCH = 3
    BATCH_SIZE = 32  # 16 / 8 / 4 / 2 / 1
    LR = 0.001
    log_interval = 2
    val_interval = 1
    classes = 2
    start_epoch = -1
    lr_decay_step = 1

    # ============================ step 1/5 数据 ============================
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((256)),  # (256, 256) 区别
        transforms.CenterCrop(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    # single-scale
    normalizes = transforms.Normalize(norm_mean, norm_std)
    valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.TenCrop(224, vertical_flip=False),
        transforms.Lambda(lambda crops: torch.stack([normalizes(transforms.ToTensor()(crop)) for crop in crops])),
    ])

    # 构建MyDataset实例
    train_data = CatDogDataset(data_dir=data_dir, mode="train", transform=train_transform)
    valid_data = CatDogDataset(data_dir=data_dir, mode="valid", transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=4)

    # ============================ step 2/5 模型 ============================
    vgg16_model = get_vgg16(path_state_dict, device, False)

    num_ftrs = vgg16_model.classifier._modules["6"].in_features
    vgg16_model.classifier._modules["6"] = nn.Linear(num_ftrs, num_classes)

    vgg16_model.to(device)
    # ============================ step 3/5 损失函数 ============================
    criterion = nn.CrossEntropyLoss()
    # ============================ step 4/5 优化器 ============================
    # 冻结卷积层
    flag = 0
    # flag = 1
    if flag:
        fc_params_id = list(map(id, vgg16_model.classifier.parameters()))  # 返回的是parameters的 内存地址
        base_params = filter(lambda p: id(p) not in fc_params_id, vgg16_model.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': LR * 0.1},  # 0，其实这里直接把grad设为true更好
            {'params': vgg16_model.classifier.parameters(), 'lr': LR}], momentum=0.9)

    else:
        optimizer = optim.SGD(vgg16_model.parameters(), lr=LR, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)

    # ============================ step 5/5 训练 ============================
    train_curve = list()
    valid_curve = list()

    for epoch in range(start_epoch + 1, MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        vgg16_model.train()
        for i, data in enumerate(train_loader):

            # forward
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = vgg16_model(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().cpu().sum().numpy()

            # 打印训练信息
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i + 1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} lr:{}".format(
                    epoch, MAX_EPOCH, i + 1, len(train_loader), loss_mean, correct / total, scheduler.get_last_lr()))
                loss_mean = 0.

        scheduler.step()  # 更新学习率

        # validate the model
        if (epoch + 1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            vgg16_model.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    bs, ncrops, c, h, w = inputs.size()
                    outputs = vgg16_model(inputs.view(-1, c, h, w))
                    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

                    loss = criterion(outputs_avg, labels)

                    _, predicted = torch.max(outputs_avg.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

                    loss_val += loss.item()

                loss_val_mean = loss_val / len(valid_loader)
                valid_curve.append(loss_val_mean)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, j + 1, len(valid_loader), loss_val_mean, correct_val / total_val))
            vgg16_model.train()

    train_x = range(len(train_curve))
    train_y = train_curve

    train_iters = len(train_loader)
    valid_x = np.arange(1, len(
        valid_curve) + 1) * train_iters * val_interval  # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.savefig(os.path.join(log_dir, "loss_curve.png"))
    # plt.show()