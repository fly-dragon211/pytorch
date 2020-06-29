# 在typora这里写之后复制到简书上

# 1. torchvision

## 1.1 torchvision.transforms.Compose(transforms)

把几个转换组合

![compose.transform](https://img-blog.csdnimg.cn/20191124102915236.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ0NzYxNDgw,size_16,color_FFFFFF,t_70)

## 1.2 transforms.RandomResizedCrop

T.RandomResizedCrop(n)将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小（即先随机采集，然后对裁剪得到的图像缩放为同一大小）

该操作的含义：即使只是该物体的一部分，我们也认为这是该l类物体

比如 猫的图片别裁剪缩放后，仍然认为这是一个猫

参考：https://blog.csdn.net/qq_32425195/article/details/84998030



# 2. torch.nn

## 2.1 torch.nn.Conv2d()
CLASS

torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

```py
import torch

x = torch.randn(2,1,7,3)
conv = torch.nn.conv2d(1,8,(2,3))
res = conv(x)

print(res.shape)    # shape = (2, 8, 6, 1)
```
**输入**

x
[ batch_size, channels, height_1, width_1 ]
batch_size 一个batch中样例的个数       2
channels 通道数，也就是当前层的深度 1
height_1, 图片的高                                 7
width_1, 图片的宽                                  3
————————————————
Conv2d的参数
[ in_channels, out_channels, (height_2, width_2) ]

channels, 通道数，和上面保持一致，也就是当前层的深度  1
output 输出的深度                                                                 8
height_2, 过滤器filter的高                                                      2
width_2, 过滤器filter的宽                                                       3

如果`padding`不是0，会在输入的每一边添加相应数目0
————————————————

**输出：**

res
[ batch_size,output, height_3, width_3 ]

batch_size, 一个batch中样例的个数，同上           2
output 输出的深度                                                  8
height_3, 卷积结果的高度                                      6 = height_1 - height_2 + 1 = 7-2+1
width_3, 卷积结果的宽度                                       1 = width_1 - width_2 +1 = 3-3+1
**如果使用padding，则height_3, width_3重新计算**
————————————————

**例子**

```
>>> # With square kernels and equal stride
>>> m = nn.Conv2d(16, 33, 3, stride=2)
>>> # non-square kernels and unequal stride and with padding
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
>>> # non-square kernels and unequal stride and with padding and dilation
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
>>> input = torch.randn(20, 16, 50, 100)
>>> output = m(input)
```
## 2.2 torch.nn.MaxPool2d

*CLASS*  `torch.nn.MaxPool2d` (*kernel_size*, *stride=None*, *padding=0*, *dilation=1*, *return_indices=False*, *ceil_mode=False*)

一般只写kernel_size，如果为2，则$H_{out} = H_{in}/2$。

## 2.3 nn.CrossEntropyLoss()

https://blog.csdn.net/geter_CS/article/details/84857220?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase



# 3 torch.nn.functional

## 3.1 torch.nn.functional.relu

Applies the rectified linear unit function element-wise:
$ReLU(x)=max(0,x)$

## 3.2 torch.nn.functional.max_pool2d(*args, **kwargs)

Applies a 2D max pooling over an input signal composed of several input planes.

torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False) 

max_pool2d(x, 2, 2) 就是H, W都除以2



# 4. torch

## 4.1 torch.max

`torch.max`(*input*, *dim*, *keepdim=False*, *out=None) -> (Tensor*, *LongTensor*)

Returns a namedtuple `(values, indices)` where `values` is the maximum value of each row of the `input` tensor in the given dimension `dim`. And `indices` is the index location of each maximum value found (argmax).

### example

~~~python
>>> a = torch.randn(4, 4)
>>> a
tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
        [ 1.1949, -1.1127, -2.2379, -0.6702],
        [ 1.5717, -0.9207,  0.1297, -1.8768],
        [-0.6172,  1.0036, -0.6060, -0.2432]])
>>> torch.max(a, 1)
torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1]))
~~~



## 4.2 in-place

~~~python
in-place operation 在 pytorch中是指改变一个tensor的值的时候，不经过复制操作，而是在运来的内存上改变它的值。可以把它称为原地操作符。
 
在pytorch中经常加后缀 “_” 来代表原地in-place operation, 比如 .add_() 或者.scatter() 
python 中里面的 += *= 也是in-place operation。
 
 
下面是正常的加操作,执行结束加操作之后x的值没有发生变化：
import torch
x=torch.rand(2) #tensor([0.8284, 0.5539])
print(x)
y=torch.rand(2)
print(x+y)      #tensor([1.0250, 0.7891])
print(x)        #tensor([0.8284, 0.5539])
 
 
下面是原地操作，执行之后改变了原来变量的值：
import torch
x=torch.rand(2) #tensor([0.8284, 0.5539])
print(x)
y=torch.rand(2)
x.add_(y)
print(x)        #tensor([1.1610, 1.3789])
~~~



# 5. 一些torch的坑

## 5.1 tensor除法

先看一段代码

~~~python
a = torch.tensor(3)
b = 2
c = a / b
e = 3

print(c, e / b)
"""
tensor(1) 1.5
"""
~~~

如果是tensor和一个整数相除，结果为除数(整数)，如果想得到小数，有下面几种方法：

1. 使用``.item``取出数。
2. 把整数变成2.0，也就是把除数变成float。

~~~python
a = torch.tensor(3)
b = 2
c = a / 2.0
e = a / float(b)
print(c, e)
~~~



参考：

[torch.nn.Conv2d](https://pytorch.org/docs/master/nn.html?highlight=conv2d#torch.nn.Conv2d)
[torch.nn.MaxPool2d](https://pytorch.org/docs/master/nn.html?highlight=maxpool2d#torch.nn.MaxPool2d)