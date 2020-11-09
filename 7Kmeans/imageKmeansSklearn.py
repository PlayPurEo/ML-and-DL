# author : 'wangzhong';
# date: 09/11/2020 20:41

from scipy.io import loadmat
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = loadmat("bird_small.mat")
A = data['A']
B = A[::3, ::3]
plt.subplot(1,2,1)
plt.imshow(A)
plt.subplot(1,2,2)
plt.imshow(B)
plt.show()

X = B.reshape(-1, 3)
mod = KMeans(n_clusters=16)
labels = mod.fit_predict(X)
colors = mod.cluster_centers_
final = colors[labels]

plt.imshow(final.reshape(43,43,3) / 255)
plt.show()
print(sys.getsizeof(A / 255))
print(sys.getsizeof(B / 255))
print(sys.getsizeof(final))