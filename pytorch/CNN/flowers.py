# author : 'wangzhong';
# date: 07/01/2021 17:16

"""
2.用pytorch进行迁移学习
图像100分类实战
流程：
1. data argumentation
2. dataloader，做批量处理
"""
import json

import torch
from torch import nn
import torch.optim as optim
import torchvision
import os
from torchvision import transforms, models, datasets
import numpy as np
import matplotlib.pyplot as plt


def flower_start():
    data_dir = './flower_data/'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    # 读取对应的花名
    with open("cat_to_name.json") as f:
        cat_to_name = json.load(f)

    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
                                     transforms.CenterCrop(224),  # 从中心开始裁剪，只得到一张图片
                                     transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 概率为0.5
                                     transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                                     transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                                     # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
                                     transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
                                     transforms.ToTensor(),
                                     # 迁移学习，用别人的均值和标准差
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
                                     ]),
        'valid': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     # 预处理必须和训练集一致
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),
    }

    batch_size = 8

    # train和valid的图片，做transform之后用字典保存
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                      ['train', 'valid']}
    # 批量处理，这里都是tensor格式（上面compose）
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
                   ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    print(dataset_sizes)
    # 样本数据的标签
    class_names = image_datasets['train'].classes
    # print(class_names)

    # 随便画一下，看一下处理后的图像
    # fig = plt.figure(figsize=(20, 12))
    # columns = 4
    # rows = 2
    # dataiter = iter(dataloaders['valid'])
    # inputs, classes = next(dataiter)
    # for idx in range(columns * rows):
    #     ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
    #     # classes为索引，class_name里为实际label，再去拿到对应的花名
    #     ax.set_title(cat_to_name[str(int(class_names[classes[idx]]))])
    #     img = transforms.ToPILImage()(inputs[idx])
    #     plt.imshow(img)
    # plt.show()


if __name__ == '__main__':
    flower_start()