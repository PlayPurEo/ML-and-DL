# author : 'wangzhong';
# date: 28/01/2021 19:07
import json
import os
import time
from matplotlib import pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from dlmodel.resnet.tools.common_tools import get_resnet_50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    # hard code
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # path --> img
    img_rgb = Image.open(path_img).convert('RGB')

    # img --> tensor
    img_tensor = img_transform(img_rgb, inference_transform)
    img_tensor.unsqueeze_(0)        # chw --> bchw
    img_tensor = img_tensor.to(device)

    return img_tensor, img_rgb


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


if __name__ == '__main__':
    # config
    path_state_dict_18 = os.path.join("..", "data", "resnet18-5c106cde.pth")
    path_state_dict_50 = os.path.join("..", "data", "resnet50-19c8e357.pth")
    path_img = os.path.join("..", "data", "Golden Retriever from baidu.jpg")
    # path_img = os.path.join(BASE_DIR, "..", "data", "tiger cat.jpg")
    path_classnames = os.path.join("..", "data", "imagenet1000.json")
    path_classnames_cn = os.path.join("..", "data", "imagenet_classnames.txt")

    # load class names
    cls_n, cls_n_cn = load_class_names(path_classnames, path_classnames_cn)

    # 1/5 load img
    img_tensor, img_rgb = process_img(path_img)

    # 2/5 load model
    resnet_model = get_resnet_50(path_state_dict_18, device, True)

    # 3/5 inference  tensor --> vector
    with torch.no_grad():
        time_tic = time.time()
        outputs = resnet_model(img_tensor)
        time_toc = time.time()

    # 4/5 index to class names
    _, pred_int = torch.max(outputs.data, 1)
    _, top5_idx = torch.topk(outputs.data, 5, dim=1)

    pred_idx = int(pred_int.cpu().numpy())
    pred_str, pred_cn = cls_n[pred_idx], cls_n_cn[pred_idx]
    print("img: {} is: {}\n{}".format(os.path.basename(path_img), pred_str, pred_cn))
    print("time consuming:{:.2f}s".format(time_toc - time_tic))

    # 5/5 visualization
    plt.imshow(img_rgb)
    plt.title("predict:{}".format(pred_str))
    top5_num = top5_idx.cpu().numpy().squeeze()
    text_str = [cls_n[t] for t in top5_num]
    for idx in range(len(top5_num)):
        plt.text(5, 15 + idx * 30, "top {}:{}".format(idx + 1, text_str[idx]), bbox=dict(fc='yellow'))
    plt.show()