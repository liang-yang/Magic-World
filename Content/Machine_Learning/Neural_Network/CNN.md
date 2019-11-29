<!-- toc -->

# CNN

CNN，Convolution Neural Network，卷积神经网络，是目前 图像领域 最好的 特征提取 方法。

## 1. Network Structure

CNN 的输入为 图像的像素及颜色通道，输出为 图像的特征空间。 图像分类时，把 CNN 输出的特征空间作为 全连接神经网络(fully connected neural network, FCN)的输入，用 FCN 完成图像分类。

深层的卷积神经网络（CNNs，请注意这里是复数）通常由多个 CNN 层层连接组成，我们称这些 CNN 处于不同阶段（**stage**）。不同 stage 里 CNN 可以存在不同的结构。

基础的 CNN 由 卷积（**convolution**），激活（**activation**），池化（**pooling**） 三种结构组成。从 特征空间 的角度，可以认为：

1. 原始图像输入为 input maps volume；
2. 卷积 和 激活 后转化为 feature maps volume；
3. 池化 后转化为 pooled maps volume。

<div align=center>![](https://tva1.sinaimg.cn/large/006y8mN6gy1g9dmgahwkdj30hs0net9v.jpg)

## 2. Convolution

### 2.1. Kernel

卷积 的目的是重点挖掘相邻元素间的关系。 具体到网络中，卷积是一种运算，是将输入空间的子矩阵与权重矩阵数乘后之和作为输出空间的元素。

<div align=center>![](https://tva1.sinaimg.cn/large/006y8mN6gy1g9dq8b30rcj30hs0cndgb.jpg)

> 卷积结果为 $$105*0+102*(-1)+100*0+103*(-1)+99*5+103*(-1)+101*0+98*(-1)+104*0=89$$。

上图中的权重矩阵我们称为 卷积核（**kernel**），有时也称作 滤波器（**filter**）。将 卷积核 在 输入空间 上平移就得到输出空间。 

可以看出，kernel的参数个数远小于 DNN。 我们假设输入空间大小为 5\*5，输出空间大小为 3\*3。那么在 CNN 网络中需要 3\*3 卷积核，参数个数为 9。在 DNN 网络中则需要 5\*5\*3\*3 = 225 个参数。

### 2.2. Padding

仔细观察会发现，卷积核使得会 输出空间 小于 输入空间，将输入图像的边缘被“修剪”掉了。 这是因为边缘上的像素永远不会位于卷积核中心，而卷积核也没法扩展到边缘区域以外。 

在部分场景下，我们会希望边缘像素不被修剪掉。**Padding** 就是针对这个问题提出的一个解决方案：它会用额外的 0 像素填充边缘。这样，当卷积核扫描输入数据时，它能延伸到边缘以外的 0 像素，扫描到边缘像素。

### 2.3. Stride

平移卷积核时，会先从输入的左上角开始，每次往左滑动一列或者往下滑动一行逐一计算输出。 我们将每次滑动的行数和列数称为 **Stride**，也称为步长。 

Stride 的作用是成倍缩小尺寸，而这个参数的值就是缩小的具体倍数，比如步幅为2，输出就是输入的1/2；步幅为3，输出就是输入的1/3。

### 2.4. Bias

同 DNN 一样，除了权重矩阵，还需要 偏移量（Bias）。

在 CNN 里，一个卷积 kernel 固定的包含一个 Bias，不管其输入的是单通道还是多通道。

因此，卷积核的输出元素为 输入矩阵 与 权重矩阵 数乘后之和再与 Bias 相加的结果。

### 2.5. Multi Input Channel

大多数输入图像都有 RGB 3 个通道：

<div align=center>![](https://tva1.sinaimg.cn/large/006y8mN6gy1g9dsw7as44j30k0044dg3.jpg)

根据上述分析，一个输入通道通过一个 2-D 卷积核生成输出空间的一个元素。因此，多个输入通道需要对应数量的 2-D 卷积核，将它们的结果线性相加，再加上 Bias，最终得到输出空间的元素。

<div align=center>![](https://tva1.sinaimg.cn/large/006y8mN6gy1g9dusnshzuj30lq0iu0u9.jpg)

### 2.6. Multi Output Channel

以输入单通道为例。 一个卷积kernel会生成一个结果矩阵，即输出单通道。 那么，经过多个卷积kernel，就会生成多个结果矩阵，就称为输出多通道，可以认为抓取了多个特征，得到多个 feature map。

### 2.7. Parameter Number

假设输入图像大小：$$w_{in} * h_{in} * c_{in}$$，卷积核大小为 $$w_{conv}*h_{conv}$$，输出通道数量为 $$c_{out}$$，则此卷积层的参数数量为：

$$
(w_{conv}*h_{conv}*c_{in} + 1) * c_{out}
$$

> width，height，channel (也作 depth)。

## 3. Activation

与 DNN 类似，CNN 依然需要通过激活处理，不再赘述。

## 4. Pooling

池化，是一种 降采样（Subsample）操作，主要目标是降低 feature maps 的特征空间，更便于高层特征的抽取。 这种暴力降低在计算力足够的情况下是不是必须的，并不确定。目前一些大的 CNNs 网络只是偶尔使用 pooling。

主要的池化方式有：

- **max pooling**；
- **average pooling**；
- **L2 pooling**；

<div align=center>![](https://tva1.sinaimg.cn/large/006y8mN6gy1g9dweug4fgj30e009zglx.jpg)

## 5. Understand

> Reference    
> -- [卷积神经网络工作原理的直观解释](https://www.zhihu.com/question/39022858)

我们将卷积 kernel 称作 滤波器，是由于权重矩阵具有过滤功能。

下图中，右矩阵为卷积kernel，其在图像中展示为一条竖弯。kernel在输入图像中平移时，如果图像与其类似，输出的元素值就会很大，否则就会很小。

<div align=center>![](https://tva1.sinaimg.cn/large/006y8mN6gy1g9dx1165j5j30jl079wf2.jpg)

<div align=center>![](https://tva1.sinaimg.cn/large/006y8mN6gy1g9dx19mshpj30ji06zmxo.jpg)

卷积神经网络的第一个卷积层的滤波器用来检测低阶特征，比如边、角、曲线等。随着卷积层的增加，对应滤波器检测的特征就更加复杂。比如第二个卷积层的输入实际上是第一层的输，用来检测低阶特征的组合（半圆、四边形等），如此累积，以检测越来越复杂的特征。

例如，下面简单的通过 4 个卷积核识别 “人”：

<div align=center>![](https://tva1.sinaimg.cn/large/006y8mN6gy1g9dxe4dxhqj30fc0cijs4.jpg)

## Reference

- [CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/)
- [卷积神经网络](https://www.zhihu.com/topic/20043586/intro)
- [绝妙可视化：什么是深度学习的卷积？](https://zhuanlan.zhihu.com/p/42090228)
- [卷积神经网络](https://www.zybuluo.com/hanbingtao/note/485480)






