<!-- toc -->

# Loss Function

---

## 1. Zero One Loss

0-1损失函数是最为简单的一种损失函数，多适用于分类问题中，如果预测值与目标值不相等，说明预测错误，输出值为1；如果预测值与目标值相同，说明预测正确，输出为0，言外之意没有损失。

sklearn 中，可通过函数 [sklearn.metrics.zero_one_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.zero_one_loss.html) 计算 0-1损失函数：

{%ace edit=true, lang='java'%}
from sklearn.metrics import zero_one_loss
import numpy as np

print("multiclass loss rate: ", zero_one_loss([1, 2, 3, 4], [2, 2, 3, 4]))
// multiclass loss rate:  0.25
print("multiclass loss count: ", zero_one_loss([1, 2, 3, 4], [2, 2, 3, 4], normalize=False))
// multiclass loss count:  1
print("multilabel loss rate: ", zero_one_loss(np.array([[0, 1], [1, 1], [1, 0]]), np.array([[1, 1], [1, 1], [1, 1]])))
// multilabel loss rate:  0.6666666666666667
{%endace%}

> 需要注意在 multilabel 场景下时，0-1损失函数 需要所有分量都相等才会判定为相等。

由于0-1损失函数过于理想化、严格化，且数学性质不是很好，难以优化，所以在实际问题中很少使用。

## 2. Hamming Loss

Hamming Loss（汉明距离）是指两个分类结果中不同分类的样本比例。Hamming Loss越小，表明分类结果差异越小。

sklearn 中，可通过函数 [sklearn.metrics.hamming_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html) 计算 Hamming Loss：

{%ace edit=true, lang='java'%}
from sklearn.metrics import hamming_loss
import numpy as np

print("multiclass loss rate: ", hamming_loss([1, 2, 3, 4], [2, 2, 3, 4]))
// multiclass loss rate:  0.25
print("multilabel loss rate: ", hamming_loss(np.array([[0, 1], [1, 1], [1, 0]]), np.array([[1, 1], [1, 1], [1, 1]])))
// multilabel loss rate:  0.3333333333333333
{%endace%}

> 可以看出，在 multilabel 场景下时，Hamming Loss 是计算不等的分量的比例，与 0-1损失函数 显著不同

see [Hamming-Loss计算公式](http://sofasofa.io/forum_main_post.php?postid=1000563)

## 3. Log Loss

对数损失函数也是常见的一种损失函数，基于概率估计而定义，常用于逻辑回归问题中。

在 Log Loss 中，输入参数为预测的各种分类的概率值，逻辑为正确分类的概率值越高，Log Loss越小。其计算公式为：

$$
L(Y, P(Y)) = - \frac{1}{N} \sum^{N}_{i=1} \log p_i
$$

> 上式中，$$N$$ 为样本数量，$$Y$$ 为样本的真实分类，$$P(Y)$$ 为预测的样本各个分类的概率值，$$p_i$$ 为预测正确样本分类的概率值。

逻辑回归使用 Log Loss 作为损失函数，是由于其目标是求

$$
\prod_{i=1}^N p_i
$$

的最大值，而根据 最大似然估计 的方法，取对数，即为求

$$
\sum^{N}_{i=1} \log p_i
$$

最大。再乘以 -1，则转化为求 Log Loss 最小。

sklearn 中，可通过函数 [sklearn.metrics.log_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html) 计算 对数损失函数：

{%ace edit=true, lang='java'%}
from sklearn.metrics import log_loss
import numpy as np

y_true = [0, 0, 1, 1, 2, 2]
y_pred = [[.8, .1, .1], [.8, .2, 0], [.3, .6, .1], [.01, .98, .01], [.1, .08, .82], [.1, .17, .73]]

print('Log Loss: ', log_loss(y_true, y_pred))
// Log Loss:  0.24841285287924483
print('Formula: ', (np.log(0.8) + np.log(0.8) + np.log(0.6) + np.log(0.98) + np.log(0.82) + np.log(0.73)) / -6)
// Formula:  0.24841285287924467
{%endace%}

## 4. Hinge Loss

Hinge Loss 是 SVM 常用的一种损失函数。考虑 SVM 的场景，当超平面可以正确分类，并且分类距离大于1时，就认为损失函数为0，否则就认为损失函数大于0。 公式为：

$$
Hinge Loss = \max(0, 1 - \hat{y} \cdot y)
$$

其中，$$\hat{y}$$ 表示样本点与分类超平面的预测距离，$$y$$ 为真实的分类（一般为 -1,1 的二分类）：
- 如果 $$\hat{y} \cdot y < 1$$，则损失为：$$1 - \hat{y} \cdot y$$；
- 如果 $$\hat{y} \cdot y >= 1$$，则损失为：$$0$$；

Hinge Loss 的图像为：
![](http://ww4.sinaimg.cn/large/006tNc79gy1g51q4cxpd2j30fz0bngln.jpg)
此即为 Hinge（合页）的由来。

sklearn 中，可通过函数 [sklearn.metrics.hinge_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hinge_loss.html) 计算 Hinge Loss：

{%ace edit=true, lang='java'%}
from sklearn.metrics import hinge_loss

print(hinge_loss([-1, 1, 1], [-2.18173682, 2.36360149, 0.09093234]))
// 0.30302255333333333
{%endace%}

