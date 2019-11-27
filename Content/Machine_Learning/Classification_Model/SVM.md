<!-- toc -->

# SVM




Hinge 损失函数是 SVM 常用的损失函数。考虑 SVM 的场景，当超平面可以正确分类，并且分类距离大于1时，就认为损失函数为0，否则就认为损失函数大于0。 公式为：

$$
Hinge Loss = \max(0, 1 - \hat{y} \cdot y)
$$

其中，$$\hat{y}$$ 表示样本点与分类超平面的预测距离，$$y$$ 为真实的分类（一般为 -1,1 的二分类）：
- 如果 $$\hat{y} \cdot y < 1$$，则损失为：$$1 - \hat{y} \cdot y$$；
- 如果 $$\hat{y} \cdot y >= 1$$，则损失为：$$0$$；

Hinge Loss 的图像为：
<div align=center>![](https://tva1.sinaimg.cn/bmiddle/006tNc79gy1g51q4cxpd2j30fz0bngln.jpg)
此即为 Hinge（合页）的由来。