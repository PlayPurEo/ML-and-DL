# author : 'wangzhong';
# date: 25/01/2021 20:09
import torch
from torchvision import models


def get_vgg16(path_state_dict, device, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = models.vgg16()
    pretrained_state_dict = torch.load(path_state_dict)
    model.load_state_dict(pretrained_state_dict)
    model.eval()

    # if vis_model:
    #     from torchsummary import summary
    #     summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model