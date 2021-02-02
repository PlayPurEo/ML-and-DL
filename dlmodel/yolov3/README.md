# YOLOV3-Pytorch
self-learning

## Author
wangzhong

---
### 1.数据准备
数据集为voc2007 train+test
* 下载地址：
    * [voc2007train](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
    * [voc2007test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)
* voc2yolo.py 进行train.txt的生成，总共是9900多样本，我取了0.4，共3985个图片作为train
* voc_annotation.py  进行实际数据的生成，格式为图片路径+gt的位置信息+类别，供dataset读取
* _训练时间：本人非迁移训练，从头训练，100个epoch，每个epoch差不多5min，GPU为1070_

### 2.model
* backbone为darknet53
* 整体yolov3架构在yolov3.py中

### 3.dataset
* 在utils/dataloader.py中，对图像进行了缩放，最后为416*416，并将gt box的坐标进行了相应的调整
    调整为(x,y,w,h)，对应416*416做了归一化
  
### 4.损失函数
* 设置三个loss对象，每个负责一个特征图
* loss里，首先将anchors缩放到该特征图下的宽和高，然后将anchors和gt boxes进行iou计算，选出
    对应位置的iou_max的anchor，同时设置好gtbox对应它的anchor的偏移值，位置的四个值。还有conf，cls_conf
    。另外设置mask和noobj_mask，方便筛选该参加损失计算的anchor box
* 对于与gtbox的iou很大但不是最大的，不设置为负样本，不参与任何的损失计算
* 损失包括四个位置的损失，conf的损失（这里有负样本），还有一个类别的损失

### 5.训练
* 自行设置batch size和epoch，视情况而定。也可迁移他人的预训练模型。这里我没有上传自己的训练模型，大小限制。

### 6.inference
* inference.py文件，会输出三个特征图对应的结果矩阵，并进行过滤和nms，最后会画出图像。
### 7.结果
* 这里暂时没有编写计算mAP的代码，直观地查看测试图片的目标定位结果，还是很不错的。