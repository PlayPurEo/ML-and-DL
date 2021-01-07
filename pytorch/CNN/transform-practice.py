# author : 'wangzhong';
# date: 07/01/2021 17:55

"""
pytorch-transform练习
"""

from PIL import Image
from torchvision import transforms, datasets
import torch
import matplotlib.pyplot as plt


def prac():
    image = Image.open("./flower_data/train/1/image_06734.jpg")
    # 原图为W*H
    plt.imshow(image)
    plt.show()

    input_trans = transforms.Compose([
        transforms.Resize(256)
    ])

    image = input_trans(image)

    print(image.size)
    plt.imshow(image)
    plt.show()


def batch_prac():
    input_trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    imgSet = datasets.ImageFolder("./flower_data/train", input_trans)
    # imgSet[0]的数据类型为tuple, (tensor, int)
    # tensor为图像transform之后的数据，int为label的索引
    # transforms.ToPILImage()(imgSet[0][0]).show()
    imagLoader = torch.utils.data.DataLoader(imgSet, batch_size=10, shuffle=True)
    for data in imagLoader:
        input, labels = data
        print(labels)
        print(input.shape)
        # nums即为batch size的大小
        nums = input.shape[0]
        for i in range(nums):
            image_ori = input[i].squeeze(0)
            print(image_ori.shape)
            # 转为原始图片
            img = transforms.ToPILImage()(image_ori)
            img.show()
            break
        break


if __name__ == '__main__':
    batch_prac()