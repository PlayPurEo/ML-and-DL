# author : 'wangzhong';
# date: 14/01/2021 23:06

"""
简单的GAN网络理解
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # 将module以列表形式传入，前面加*号，相当于把列表里的module当做一个个参数传入
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


def get_device() -> torch.device:
    # 是否用GPU训练
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def start():
    generator = Generator()
    discriminator = Discriminator()
    adversarial_loss = torch.nn.BCELoss()
    device = get_device()
    generator.to(device)
    discriminator.to(device)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=128,
        shuffle=True,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    epochs = 100
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            valid = torch.ones((imgs.size(0), 1), requires_grad=False)
            fake = torch.zeros((imgs.size(0), 1), requires_grad=False)

            rand_image = torch.randn((imgs.size(0), 100))
            rand_image.to(device)
            fake_imgs = generator(rand_image)

            optimizer_G.zero_grad()
            # 生成器就是让生成的图片更倾向于真，所以这里的损失函数是与valid进行对比
            g_loss = adversarial_loss(discriminator(fake_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(imgs), valid)
            fake_loss = adversarial_loss(discriminator(fake_imgs), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.zero_grad()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % 400 == 0:
                save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)


if __name__ == '__main__':
    # print(
    #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
    #     % (1, 100, 10, 100, 5.00, 8.00)
    # )
    print("[Epoch {:.0f}/{:.0f}] [Batch {:.0f}/{:.0f}] [D loss: {:.3f}] [G loss: {:.3f}]".format(1, 100, 10, 100, 5, 8))