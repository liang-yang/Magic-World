<!-- toc -->

# Discretization

---

数据 **离散化**，是指将连续数据转化为离散数据。

- 示例1：学生考试成绩(100满分)，我们将成绩介于90-100的评为优秀，80-89的评为良好，70-79的评为中等，60-69的评为及格，60以下的评为不及格；
- 示例2：机器学习中逻辑回归算法可用作分类，实际就是将【0,1】之间的连续数据（一般是概率），通过阈值转化为离散数据 0 和 1；

数据离散化的好处，主要有：

1. 减小计算量，运算速度加快；
> 以示例1中学生考试成绩为例，假设1千万个学生需要存储1千万个数据，但通过离散化，仅需存储5个数据即可衡量总体考试水平。
2. 模型更加健壮；
> 因为数据在一定范围内波动，并不会影响最终的结果，在一定程度上降低了过拟合的风险。

## 1. k-bins discretization

k-bins discretization 是指将某特征划分为 k 份（k个箱子），具体的划分方式有 三 种：

1. **uniform**：所有箱子具有相同的宽度；
2. **quantile**：所有箱子包含相同的样本数量；
3. **kmeans**：基于 k-means 聚类算法划分；

在 sklearn 中可通过 [sklearn.preprocessing.KBinsDiscretizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html) 实现k-bins离散。

{%ace edit=false, lang='java'%}
from sklearn.preprocessing import KBinsDiscretizer

X = [[-2, 10, -4, -1],
     [-1, 1, -3, -0.5],
     [0, 3, -2, 0.5],
     [1, 4, -1, 2],
     [3, 2, 0, 1]]
est = KBinsDiscretizer(n_bins=[3, 3, 2, 2], encode='ordinal', strategy='uniform')
Xt = est.fit_transform(X)
print(est.bin_edges_)
// [array([-2.        , -0.33333333,  1.33333333,  3.        ])
//  array([ 1.,  4.,  7., 10.]) array([-4., -2.,  0.])
//  array([-1. ,  0.5,  2. ])]
print(Xt)
// [[0. 2. 0. 0.]
//  [0. 0. 0. 0.]
//  [1. 0. 1. 1.]
//  [1. 1. 1. 1.]
//  [2. 0. 1. 1.]]
{%endace%}

## 2. Feature binarization

Feature binarization 是指将特征值基于 阈值 转化为布尔值。例如在文本分析中，为了简化计算，有时仅考虑 word 在文本中是否存在，而不考虑出现的频率。

在 sklearn 中可通过函数 [sklearn.preprocessing.binarize](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.binarize.html) 和类 [sklearn.preprocessing.Binarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html) 实现 feature binarization。

{%ace edit=false, lang='java'%}
from sklearn.preprocessing import Binarizer

X = [[1., -1., 2.],
     [2., 0., 0.],
     [0., 1., -1.]]
binarizer = Binarizer(threshold=1.1)
Xb = binarizer.fit_transform(X)
print(Xb)
// [[0. 0. 1.]
//  [1. 0. 0.]
//  [0. 0. 0.]]
{%endace%}







