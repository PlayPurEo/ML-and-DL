# author : 'wangzhong';
# date: 24/01/2021 02:41
import os
import random

from PIL import Image
from torch.utils.data import Dataset


class CatDogDataset(Dataset):
    def __init__(self, data_dir, mode="train", split_n=0.9, rng_seed=620, transform=None):
        """
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.mode = mode
        # ../data
        self.data_dir = data_dir
        self.rng_seed = rng_seed
        # train val 比例
        self.split_n = split_n
        self.data_info = self._get_img_info()  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        if len(self.data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.data_info)

    def _get_img_info(self):

        img_names = os.listdir(self.data_dir)
        # 获取到所有文件名
        img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

        # 固定这个shuffle，这样才能拿到不同的train和valid
        random.seed(self.rng_seed)
        random.shuffle(img_names)

        # 获取label， 二分类， 猫和狗
        img_labels = [0 if n.startswith('cat') else 1 for n in img_names]

        # 做好train和val的比例
        split_idx = int(len(img_labels) * self.split_n)  # 25000* 0.9 = 22500
        # split_idx = int(100 * self.split_n)
        if self.mode == "train":
            img_set = img_names[:split_idx]     # 数据集90%训练
            # img_set = img_names[:22500]     #  hard code 数据集90%训练
            label_set = img_labels[:split_idx]
        elif self.mode == "valid":
            img_set = img_names[split_idx:]
            label_set = img_labels[split_idx:]
        else:
            raise Exception("self.mode 无法识别，仅支持(train, valid)")

        # 拼接上每个图片的名字
        path_img_set = [os.path.join(self.data_dir, n) for n in img_set]
        data_info = [(n, l) for n, l in zip(path_img_set, label_set)]

        return data_info
