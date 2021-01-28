# author : 'wangzhong';
# date: 28/01/2021 20:47
import os

from PIL import Image
from torch.utils.data import Dataset


class CifarDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        assert (os.path.exists(data_dir)), "data_dir:{} 不存在！".format(data_dir)

        self.data_dir = data_dir
        self._get_img_info()
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.img_info[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("未获取任何图片路径，请检查dataset及文件路径！")
        return len(self.img_info)

    def _get_img_info(self):
        # 拿到分好类的文件夹
        sub_dir_ = [name for name in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, name))]
        sub_dir = [os.path.join(self.data_dir, c) for c in sub_dir_]

        self.img_info = []
        for c_dir in sub_dir:
            # basename函数拿到当前文件夹的名字，也就是label，和该label下的图片路径做一个元组存储
            path_img = [(os.path.join(c_dir, i), int(os.path.basename(c_dir))) for i in os.listdir(c_dir) if
                        i.endswith("png")]
            # 这里其实应该做一个shuffle
            self.img_info.extend(path_img)