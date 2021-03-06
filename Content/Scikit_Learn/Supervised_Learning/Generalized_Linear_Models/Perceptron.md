<!-- toc -->

# Perceptron

---

感知机是二分类的线性分类模型，在特征空间中通过一个超平面将特征向量分为两类，是神经网络和支持向量机的基础。下面按照机器学习三要素（模型集合、目标策略、学习算法）对感知机进行分析。

> see [Perceptron](http://www.hankcs.com/ml/the-perceptron.html)

## 1. 模型集合

感知机是一种线性分类模型。对于分类模型，假设输入空间（特征空间）是 $$ x \in R^n $$，输出空间为 $$ y \in \{+1,-1\} $$，那么分类模型统一的公式为：

$$
y = Classification(x)
$$

感知机模型，是通过超平面（线性模型）将所有特征向量进行分隔，在超平面两侧的特征向量分别为两类。我们知道，超平面 的表达式为：

$$
w \cdot x + b = 0
$$

其中，$$ w \cdot x $$ 表示内积，$$ w \in R^n $$ 称作权值向量，$$ b \in R $$ 称作偏置，均为模型参数。

在超平面两侧分别为两类，则分别为 $$ w \cdot x + b > 0 $$ 和 $$ w \cdot x + b < 0 $$，那么可以使用符号函数 $$ sign $$ 作为分类函数：

$$
sign(t) = \begin{cases}
+1 &\text{if } t	\geqslant 0 \\
-1 &\text{if } t < 0
\end{cases}
$$

由此得到 感知机分类模型 的完整公式：

$$
y = sign(w \cdot x + b)
$$

感知机的模型集合为特征空间中所有的线性分类模型，也可以理解为对应维度的所有超平面，我们根据 目标策略 和 学习算法 找出最合适的超平面即可。

## 2. 目标策略

按照感知机的分类目标，最直观的策略是误分类点的个数：误分类点越少感知机模型越好。但是，误分类点的个数不连续，更不可导，而损失函数我们一般希望其连续可导以便于计算。因此，替代将 **所有误分类点到超平面的总距离** 作为损失函数。

假设超平面为 $$ w \cdot x +b = 0 $$，那么点 $$ x_0 $$ 与此超平面的绝对距离为：

$$
L(w,b)_{x_0} = \frac{|w \cdot x_0 +b|}{||w||}
$$

其中 $$ ||w|| $$ 表示 $$ w $$ 的 $$ L_2 $$ 范数。在感知机中，我们仅需要判断正负及比较相对距离，因此可不除以 $$ ||w|| $$，取相对距离即可。

对于训练集 $$ T=\{(x_1,y_1),(x_2,y_2),...(x_N,y_N)\} $$，其中 $$ x_i \in R^n $$，$$ y_i \in \{+1,-1\} $$。假设 $$ M $$ 为超平面误分类点的集合，则损失函数为：

$$
L(w,b)=-\sum_{x_i \in M} y_i(w \cdot x_i + b)
$$

感知机的 目标策略 即求使得此损失函数最小的模型参数 $$ w,b $$ 。

## 3. 学习算法

按照常规的方法，求极值就是对损失函数的未知参数求导为零。但深入分析发现，感知机的损失函数有一个前提条件：误分类点的集合固定。而实际随着参数的变化，误分类点会发生变化，也就是说 **误分类点集合不固定**。因此，我们不能通过求导的方式来求极值。

此种情况下，比较适合使用 梯度下降 这种迭代的思路。

### 3.1. 原始形式

- **输入：**

1. 训练集 $$T=\{(x_1,y_1),(x_2,y_2),...(x_N,y_N)\}$$，其中 $$x_i \in R^n$$，$$y_i \in \{+1,-1\}$$；
2. 学习率 $$\eta(1 \geqslant \eta > 0)$$；

- **输出：**

1. 参数 $$ w, b $$；
2. 感知机模型 $$ y=sign(w \cdot x+b) $$；

- **算法：**

1. 随机选取初始值 $$ w_0, b_0 $$；
2. 在训练集中选取数据 $$ (x_i, y_i) $$；
3. 如果 $$ y_i(w \cdot x_i + b) \leqslant 0 $$，我们需要使其大于0。将 $$ f(w,b) = y_i(w \cdot x_i + b) $$ 视作函数，我们就通过调整参数 $$ w, b $$，使得函数值不断增大。不同的是，这里的结束点仅需要使得函数值大于0，不需要梯度为零（实际上距离没有极大值，梯度也不可能为零）。因此我们更新参数：
$$
w = w + \eta \cdot \frac{\partial \bigg( y_i(w \cdot x_i + b)\bigg)}{\partial w} = w + \eta \cdot y_i \cdot x_i
$$

$$
b = b + \eta \cdot \frac{\partial \bigg( y_i(w \cdot x_i + b)\bigg)}{\partial b} = b + \eta \cdot y_i
$$
4. 转至步骤2，直至训练集中没有误分类点。

### 3.2. 对偶形式

由于参数 $$ w,b $$ 每一轮迭代都会发生改变，导致每一轮都需要计算内积 $$ w \cdot x_i $$，当特征维度较多时效率较低，所以考虑通过其他形式提升计算性能。

从原始形式的计算方式可以看出，如果初始点 $$w_0 = 0 $$，则参数：

$$
w = \sum_{i=1}^N n_i \cdot \eta \cdot y_i \cdot x_i = \sum_{i=1}^N \alpha_i \cdot y_i \cdot x_i
$$

$$
b = \sum_{i=1}^N n_i \cdot \eta \cdot y_i = \sum_{i=1}^N \alpha_i \cdot y_i
$$

其中，$$ n_i $$ 表示某误分类点的修改次数，$$ \alpha_i = n_i \cdot \eta $$ 作为参数。

可以看出，参数 $$ w $$ 可转化为样本点特征向量的 **线性组合**，可通过矩阵的方式提升计算效率。由此，得到原始形式的对偶形式。

- **输入：**

1. 训练集 $$T=\{(x_1,y_1),(x_2,y_2),...(x_N,y_N)\}$$，其中 $$x_i \in R^n$$，$$y_i \in \{+1,-1\}$$；
2. 学习率 $$\eta(1 \geqslant \eta > 0)$$；

- **输出：**

1. 参数 $$ \alpha, b $$，其中 $$ \alpha = (\alpha_1,\alpha_2...,\alpha_N)^T $$；
2. 感知机模型：$$ y=sign(\sum_{i=1}^N \alpha_i \cdot y_i \cdot x_i \cdot x + b) $$；

- **算法：**

1. 选取初始值 $$ \alpha = (0,0,...,0), b = 0 $$；
2. 在训练集中选取数据 $$ (x_j, y_j) $$；
3. 如果 $$ y_j(\sum_{i=1}^N \alpha_i \cdot y_i \cdot x_i \cdot x_j+b) \leqslant 0 $$，更新参数：
$$
\alpha_j = \alpha_j + \eta
$$
$$
b = b + \eta \cdot y_j
$$
4. 转至步骤2，直至训练集中没有误分类点。

可以看出，对偶形式中内积运算是固定的，我们可以预先把训练集中实例间的两两内积计算出来并以矩阵的形式存储，这个矩阵就是 $$ Gram $$ 矩阵。

$$
G = [x_i \cdot x_j]_{N \times N}
$$

基于 $$ Gram $$ 矩阵运算，可以提升计算效率。

> see [感知机学习算法的对偶形式](https://www.zhihu.com/question/26526858)

## 4. 几何意义

我们以二维平面为例对几何意义进行说明。

训练集中的实例相当于平面中的点，感知机算法就相当于求一条直线，将正负两类分隔开。我们不妨假设此直线方程为：

$$
w \cdot x + b = 0
$$

其中，$$ w $$ 是直线的法向量，$$ b $$ 是原点到直线的截距（严格的说，应该称 $$ w/\|w\| $$ 为单位法向量，$$ b/\|w\| $$ 为原点到直线的物理截距）。

因此，在算法中 $$ w = w + \eta y_i x_i $$ 就可以理解为将直线的法向量往误分类点 $$ x_i $$ 倾斜， $$ b = b + \eta y_i $$ 就可以理解为将直线沿法向量往误分类点 $$ x_i $$ 移动。简单的说，就是一个调整方向，一个调整距离，最终使得直线越过该误分类点使其被正确分类。

## 5. Demo of Python

{%ace edit=true, lang='python'%}

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
auc = metrics.auc(fpr, tpr)

print("fpr:",fpr)
print("tpr:",tpr)
print("thresholds:",thresholds)
print(auc)

plt.plot(fpr,tpr,marker = 'o')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.show()

{%endace%}

执行结果如下：

> fpr: [0. 0.5 0.5 1. ]
tpr: [0.5 0.5 1. 1. ]
thresholds: [0.8 0.4 0.35 0.1 ]
auc: 0.75

![](https://ws1.sinaimg.cn/large/006tNbRwgy1fxi50wjyxhj30fy0co3yf.jpg)

