# author : 'wangzhong';
# date: 2021/2/2 2:28
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

from dlmodel.yolov3.utils.config import Config
from dlmodel.yolov3.yolo import YOLO

if __name__ == '__main__':
    yolo = YOLO()

    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        r_image = yolo.detect_image(image)
        r_image.show()
    # a = torch.ones((3,  85))
    # b = np.array(a[:, :4])
    # c = b[:, 0]
    # d = np.expand_dims(c, -1)
    # print(d.shape)
    # gws = torch.ones((4, 1))
    # after = torch.FloatTensor(gws)
    # gt_box = torch.FloatTensor(torch.cat([torch.zeros_like(gws), torch.zeros_like(gws), gws, gws], 1))
    # print(gt_box)