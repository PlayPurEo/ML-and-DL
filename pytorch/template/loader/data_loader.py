# author : 'wangzhong';
# date: 21/01/2021 18:15

"""
所需要的loader，基于base loader
"""

from torchvision import datasets, transforms
from base.base_loader import BaseDataLoader


class MnistDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=False, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)