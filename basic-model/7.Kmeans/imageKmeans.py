# author : 'wangzhong';
# date: 08/11/2020 14:18

from scipy.io import loadmat
from TDkmeans import run_kmeans
from TDkmeans import init_centros
import numpy as np
import matplotlib.pyplot as plt

data = loadmat("bird_small.mat")
A = data['A']
# imshow()显示图像时对double型是认为在0~1范围内，即大于1时都是显示为白色，而imshow显示uint8型时是0~255范围
A = A / 255
# 不知道行数是多少，但是列数要求是3，可以用-1来表示
A = A.reshape(-1, 3)

idx, centros_all = run_kmeans(A, init_centros(A, k=16), iter=20)
centros_final = centros_all[-1]

im = np.zeros(A.shape)
k = 16
for i in range(k):
    im[idx == i] = centros_final[i]
im = im.reshape(128, 128, 3)
plt.imshow(im)
