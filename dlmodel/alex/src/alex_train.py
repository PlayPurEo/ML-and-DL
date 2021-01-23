# author : 'wangzhong';
# date: 24/01/2021 02:28
import os

import torch

# kaggle猫狗数据集：https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import numpy as np
from matplotlib import pyplot as plt
from dlmodel.alex.tools.dataset import CatDogDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 这里实际上跟test是同个函数，可以抽取出来放在utils
def get_model(path_state_dict, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = models.alexnet()
    pretrained_state_dict = torch.load(path_state_dict)
    model.load_state_dict(pretrained_state_dict)

    # if vis_model:
    #     from torchsummary import summary
    #     summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model


if __name__ == '__main__':
    data_dir = os.path.join("..", "data", "train")
    path_state_dict = os.path.join("..", "data", "alexnet-owt-4df8aa71.pth")
    num_classes = 2

    MAX_EPOCH = 3  # 可自行修改
    BATCH_SIZE = 128  # 可自行修改
    LR = 0.001  # 可自行修改
    # 多久打印一次情况
    log_interval = 1  # 可自行修改
    # 多久valid一下
    val_interval = 1  # 可自行修改
    classes = 2
    start_epoch = -1
    lr_decay_step = 1  # 可自行修改

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

    normalizes = transforms.Normalize(norm_mean, norm_std)
    valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.TenCrop(224, vertical_flip=False),
        # (10, B, C, H, W)
        transforms.Lambda(lambda crops: torch.stack([normalizes(transforms.ToTensor()(crop)) for crop in crops])),
    ])

    # 构建MyDataset实例
    train_data = CatDogDataset(data_dir=data_dir, mode="train", transform=train_transform)
    valid_data = CatDogDataset(data_dir=data_dir, mode="valid", transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=4)

    # ============================ step 2/5 模型 ============================
    alexnet_model = get_model(path_state_dict, False)

    # 迁移学习微调输出层
    num_ftrs = alexnet_model.classifier._modules["6"].in_features
    alexnet_model.classifier._modules["6"] = nn.Linear(num_ftrs, num_classes)
    alexnet_model.to(device)
    # ============================ step 3/5 损失函数 ============================
    criterion = nn.CrossEntropyLoss()
    # ============================ step 4/5 优化器 ============================
    # 冻结卷积层
    flag = 0
    # flag = 1
    if flag:
        fc_params_id = list(map(id, alexnet_model.classifier.parameters()))  # 返回的是parameters的 内存地址
        base_params = filter(lambda p: id(p) not in fc_params_id, alexnet_model.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': LR * 0.1},  # 0，其实这里直接把grad设为true更好
            {'params': alexnet_model.classifier.parameters(), 'lr': LR}], momentum=0.9)

    else:
        optimizer = optim.SGD(alexnet_model.parameters(), lr=LR, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)

    # ============================ step 5/5 训练 ============================
    train_curve = list()
    valid_curve = list()
    for epoch in range(start_epoch + 1, MAX_EPOCH):
        loss_mean = 0.
        correct = 0.
        total = 0.

        alexnet_model.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = alexnet_model(inputs)
            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 分类情况统计
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().cpu().sum().numpy()
            loss_mean = loss.item()
            train_curve.append(loss_mean)
            if (i+1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                # <左对齐，>右对齐 .4f保留四位小数
                # log_interval = 1 即每个batch都输出一次
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.
        # 一个epoch更新学习率
        scheduler.step()

        # validate the model
        if (epoch + 1) % val_interval == 0:
            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            alexnet_model.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    # [4, 10, 3, 224, 224]
                    bs, ncrops, c, h, w = inputs.size()
                    outputs = alexnet_model(inputs.view(-1, c, h, w))
                    # data argumentation后，要对输出的值做一个average
                    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

                    loss = criterion(outputs_avg, labels)

                    _, predict = torch.max(outputs_avg, dim=1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().cpu().sum().numpy()
                    loss_val += loss.item()
                loss_val_mean = loss_val / len(valid_loader)
                valid_curve.append(loss_val_mean)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, j + 1, len(valid_loader), loss_val_mean, correct_val / total_val))

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
    plt.show()