# 1. 数据

DataSet 导入数据信息；

DataLoader 导入数据，进行预处理(transform)，分配batch;

## 1.1 需要注意的地方

1. 图片是通过 Image.open()函数读取进来的,当涉及如下问题： 图片的通道顺序(RGB ? BGR ?) 图片是 `w*h*c ? c*w*h ?` 像素值范围[0-1] or [0-255] ? 就要查看 MyDataset()类中 `__getitem__()`下读取图片用的是什么方法

## 1.2 计算数据集的均值和方差

训练时依次操作

1. 随机裁剪

2. Totensor

3. 数据标准化(减均值，除以标准差)

数据标准化参数计算可以用下面的函数

```python
def get_mean_std(dataset, ratio=0.01, num_cal=1):
    """Get mean and std by sample ratio
    calculate the 'mean and std' by num_cal time and get the average
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset)*ratio), 
                                             shuffle=True, num_workers=0)
    mean = []
    std = []
    for i in range(num_cal):
        train = iter(dataloader).next()[0]   # 一个batch的数据
        mean.append(np.mean(train.numpy(), axis=(0,2,3)))
        std.append(np.std(train.numpy(), axis=(0,2,3)))
    mean = np.array(mean).mean(axis=0)
    std = np.array(std).mean(axis=0)
    return mean, std
```

下面是一个例子：

```python
# imports
import PIL
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 4
NUM_WORK = 0
data_path = r'D:\python\Pytorch\data\hymenoptera_data'
LR = 0.001
MOMENTUM = 0.9
INPUT_SIZE = 224


# transforms


# datasets
trainset = torchvision.datasets.ImageFolder(os.path.join(data_path, 'train'), 
                                            transform=transforms.Compose(
                                                [transforms.RandomResizedCrop(INPUT_SIZE),
                                                 transforms.ToTensor(),
                                                 # transforms.Normalize((0.5,), (0.5))
                                                ]
                                            ))
testset = torchvision.datasets.ImageFolder(os.path.join(data_path, 'val'), 
                                            transform=transforms.Compose(
                                                [transforms.RandomResizedCrop(INPUT_SIZE),
                                                 transforms.ToTensor(),
                                                 # transforms.Normalize((0.5,), (0.5))
                                                ]
                                            ))

# helper function to show an image
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # BRG -> RGB

def get_mean_std(dataset, ratio=0.01, num_cal=1):
    """Get mean and std by sample ratio
    calculate the 'mean and std' by num_cal time and get the average
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset)*ratio), 
                                             shuffle=True, num_workers=0)
    mean = []
    std = []
    for i in range(num_cal):
        train = iter(dataloader).next()[0]   # 一个batch的数据
        mean.append(np.mean(train.numpy(), axis=(0,2,3)))
        std.append(np.std(train.numpy(), axis=(0,2,3)))
    mean = np.array(mean).mean(axis=0)
    std = np.array(std).mean(axis=0)
    return mean, std

train_mean, train_std = get_mean_std(trainset, num_cal=100)

test_mean, test_std = get_mean_std(testset, num_cal=100)

print(train_mean, train_std)
print(test_mean,test_std)

```

## 1.2 可以对transform操作

对 transforms 操作,使数据增强更灵活 

1. transforms.RandomChoice(transforms), 从给定的一系列 transforms 中选一个进行操作 
2. transforms.RandomApply(transforms, p=0.5),给一个 transform 加上概率,依概率进行操作 
3. transforms.RandomOrder,将 transforms 中的操作随机打乱