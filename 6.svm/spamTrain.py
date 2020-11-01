# author : 'wangzhong';
# date: 01/11/2020 16:50

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.svm import SVC

data1 = loadmat('spamTrain.mat')
X, y = data1['X'], data1['y']

data2 = loadmat("spamTest.mat")
Xtest, ytest = data2['Xtest'], data2['ytest']

Cvalues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

best_score = 0
best_params = 0
for c in Cvalues:
    svc = SVC(C=c, kernel='linear')
    svc.fit(X, y.flatten())
    score = svc.score(Xtest, ytest.flatten())
    if score > best_score:
        best_score = score
        best_params = c
print(best_score, best_params)

svc_test = SVC(C=best_params, kernel='linear')
svc_test.fit(X, y.flatten())
print(svc_test.score(Xtest, ytest.flatten()))
