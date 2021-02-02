# author : 'wangzhong';
# date: 2021/2/2 15:16

import os
import random

random.seed(0)
# 采用相对路径
xmlfilepath = "Annotations"
saveBasePath = "ImageSets/Main/"
# 训练验证集的占比
trainval_percent=0.4
# 训练集在训练验证集中的占比
train_percent=1


if __name__ == '__main__':
    temp_xml = os.listdir(xmlfilepath)
    total_xml = []
    for xml in temp_xml:
        if xml.endswith(".xml"):
            total_xml.append(xml)

    num = len(total_xml)
    num_list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(num_list, tv)
    train = random.sample(trainval, tr)

    print("train and val size：", tv)
    print("train size：", tr)
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in num_list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
