# author : 'wangzhong';
# date: 18/01/2021 17:52

import torch

class Path:

    @staticmethod
    def model_dir():
        return "./models/c3d-pretrained.pth"

    @staticmethod
    def db_dir(dataset):
        root_dir = "data/UCF-101"
        output_dir = "data_process/ucf101"
        return root_dir, output_dir


if __name__ == '__main__':
    check = torch.load("run/run_0/models/C3D-ucf101_epoch-99.pth.tar", map_location=torch.device('cpu'))
    for name in check:
        print(name)