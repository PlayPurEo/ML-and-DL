# author : 'wangzhong';
# date: 2021/2/2 15:35
import torch

import numpy as np
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dlmodel.yolov3.nets.yolo_trainning import YOLOLoss
from dlmodel.yolov3.nets.yolov3 import YoloBody
from dlmodel.yolov3.utils.config import Config
from dlmodel.yolov3.utils.dataloader import YoloDataset, yolo_dataset_collate


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_ont_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_loss = 0
    val_loss = 0

    net.train()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            # 返回的是三个特征图， 13,13,75；26,26,75；52,52,75
            outputs = net(images)
            losses = []
            num_pos_all = 0
            # ----------------------#
            #   计算损失
            # ----------------------#
            # 对于三个特征图分别进行计算
            for i in range(3):
                # 这里的num pos为 batch size / 3，是因为有三个特征图损失计算
                loss_item, num_pos = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item)
                num_pos_all += num_pos

            loss = sum(losses) / num_pos_all
            # ----------------------#
            #   反向传播
            # ----------------------#
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                num_pos_all = 0
                for i in range(3):
                    loss_item, num_pos = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos
                loss = sum(losses) / num_pos_all
                val_loss += loss.item()
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    if epoch + 1 % 10 == 0:
        print('Saving state, iter:', str(epoch + 1))
        torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
        (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))


if __name__ == '__main__':
    model = YoloBody(Config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 可以选择载入预训练模型进行训练
    # model_path = "model_data/yolo_weights.pth"
    # print('Loading weights into state dict...')
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_path, map_location=device)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # print('Finished!')
    model.to(device)
    if torch.cuda.is_available():
        cudnn.benchmark = True

    # ----------------------------------------------------#
    #   获得图片路径和标签
    # ----------------------------------------------------#
    annotation_path = '2007_train.txt'
    # ----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    # ----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    lr = 1e-3
    Batch_size = 8
    Init_Epoch = 0
    Freeze_Epoch = 50

    optimizer = optim.Adam(model.parameters(), lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
    # 一张维度格式转换好的图片，和该图片上所有的gt
    train_dataset = YoloDataset(lines[:num_train], (Config["img_h"], Config["img_w"]), True)
    val_dataset = YoloDataset(lines[num_train:], (Config["img_h"], Config["img_w"]), False)
    # collate_fn表示自定义取样本的方式
    gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=2, pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=2, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)

    # 建立loss函数
    yolo_losses = []
    for i in range(3):
        # anchors : 3*3*2 三个特征图，每个元素三个anchors，一个anchors的宽和高
        yolo_losses.append(YOLOLoss(np.reshape(Config["yolo"]["anchors"], [-1, 2]),
                                    Config["yolo"]["classes"], (Config["img_w"], Config["img_h"]), True, False))

    epoch_size = num_train // Batch_size
    epoch_size_val = num_val // Batch_size
    # ------------------------------------#
    #   冻结一定部分训练
    # ------------------------------------#
    # for param in model.backbone.parameters():
    #     param.requires_grad = False

    for epoch in range(Init_Epoch, Freeze_Epoch):
        fit_ont_epoch(model, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, True)
        lr_scheduler.step()

    # ------------------------------------#
    #   解除冻结后训练
    # ------------------------------------#
    lr = 1e-4
    Batch_size = 4
    Freeze_Epoch = 50
    Unfreeze_Epoch = 100

    optimizer = optim.Adam(model.parameters(), lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    train_dataset = YoloDataset(lines[:num_train], (Config["img_h"], Config["img_w"]), True)
    val_dataset = YoloDataset(lines[num_train:], (Config["img_h"], Config["img_w"]), False)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)

    epoch_size = num_train // Batch_size
    epoch_size_val = num_val // Batch_size
    # ------------------------------------#
    #   解冻后训练
    # ------------------------------------#
    for param in model.backbone.parameters():
        param.requires_grad = True

    for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
        fit_ont_epoch(model, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda)
        lr_scheduler.step()