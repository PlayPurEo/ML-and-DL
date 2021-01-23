# author : 'wangzhong';
# date: 23/01/2021 15:53
import json
import os
import time

import torch
from PIL import Image
from torchvision import transforms, models
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_class_names(p_clsnames, p_clsnames_cn):
    """
    加载标签名
    :param p_clsnames:
    :param p_clsnames_cn:
    :return:
    """
    with open(p_clsnames, "r") as f:
        class_names = json.load(f)
    with open(p_clsnames_cn, encoding='UTF-8') as f:  # 设置文件对象
        class_names_cn = f.readlines()
    return class_names, class_names_cn


def img_transform(img_rgb, transform=None):
    """
    将数据转换为模型读取的形式
    :param img_rgb: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """

    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")

    img_t = transform(img_rgb)
    return img_t


def process_img(path_img):
    # 官方数据的mean和std，不能改
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # path --> img， 不用RGB，读出来会是4通道
    img_rgb = Image.open(path_img).convert('RGB')

    # img --> tensor
    img_tensor = img_transform(img_rgb, inference_transform)
    # 增加batch size维度
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.to(device)

    return img_tensor, img_rgb


def get_model(path_state_dict, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    # pytorch版本有一个trick，加了一个AdaptiveAvgPool2d，确定进入全连接的size为6*6
    # 这是为了防止原输入图片的size不是224*224
    model = models.alexnet()
    pretrained_state_dict = torch.load(path_state_dict)
    model.load_state_dict(pretrained_state_dict)
    model.eval()

    # if vis_model:
    #     from torchsummary import summary
    #     summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model


if __name__ == '__main__':
    # 预训练模型下载地址：http://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
    # 如果不想自己读取预训练模型，在函数里把pretrained设为true
    # kaggle猫狗数据集：https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
    path_state_dict = os.path.join("..", "data", "alexnet-owt-4df8aa71.pth")
    path_img = os.path.join("..", "data", "tiger cat.jpg")
    path_classnames = os.path.join("..", "data", "imagenet1000.json")
    path_classnames_cn = os.path.join("..", "data", "imagenet_classnames.txt")

    cls_n, cls_n_cn = load_class_names(path_classnames, path_classnames_cn)

    img_tensor, img_rgb = process_img(path_img)

    alexnet_model = get_model(path_state_dict, True)

    with torch.no_grad():
        time_tic = time.time()
        outputs = alexnet_model(img_tensor)
        time_toc = time.time()

    # top-1和top-5
    _, pred_int = torch.max(outputs.data, 1)
    _, top5_idx = torch.topk(outputs.data, 5, dim=1)

    pred_idx = int(pred_int.cpu().numpy())
    pred_str, pred_cn = cls_n[pred_idx], cls_n_cn[pred_idx]
    print("img: {} is: {}\n{}".format(os.path.basename(path_img), pred_str, pred_cn))
    print("time consuming:{:.2f}s".format(time_toc - time_tic))

    plt.imshow(img_rgb)
    plt.title("predict:{}".format(pred_str))
    top5_num = top5_idx.cpu().numpy().squeeze()
    text_str = [cls_n[t] for t in top5_num]
    for idx in range(len(top5_num)):
        plt.text(5, 15 + idx * 30, "top {}:{}".format(idx + 1, text_str[idx]), bbox=dict(fc='yellow'))
    plt.show()