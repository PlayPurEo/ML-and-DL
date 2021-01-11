# author : 'wangzhong';
# date: 11/01/2021 22:52

"""
pytorch可视化工具tensorboardX学习
"""

from tensorboardX import SummaryWriter
from PIL import Image
from torchvision import transforms

writer = SummaryWriter('runs/scalar_example')
for i in range(10):
    writer.add_scalar('quadratic', i ** 2, global_step=i)
    writer.add_scalar('exponential', 2 ** i, global_step=i)

writer.close()

writer = SummaryWriter('runs/another_scalar_example')
for i in range(10):
    writer.add_scalar('quadratic', i ** 3, global_step=i)
    writer.add_scalar('exponential', 3 ** i, global_step=i)
writer.close()

input_trans = transforms.Compose([
    transforms.ToTensor()
])
writer = SummaryWriter('runs/image_example')
for i in range(1, 3):
    writer.add_image('images',
                     input_trans(Image.open("./images/{}.jpg".format(i))),  # 注意这里传入的图片一定要是tensor格式的
                     global_step=i)
writer.close()
