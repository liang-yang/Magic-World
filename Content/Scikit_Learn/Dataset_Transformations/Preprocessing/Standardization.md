<!-- toc -->

# Standardization

---

在机器学习的算法中，大部分都需要对原始数据进行 **标准化** 处理，其目的主要有两个方面：

1. **消除量纲**。多维度的特征值结合在一起分析时，由于量纲的差异，可能导致某单个维度的特征值占据主导作用，得到不正确的结论。例如计算两个特征向量的欧式距离时，小量纲的特征值会较大，就会在距离公式中占据主导。
2. **提升效率**。例如在梯度下降方法中，对特征标准化处理会使得算法收敛效率快的多。

## 1. 标准差标准化

在统计学中，很多理论都是基于数据分布为正态分布的假设来分析的。  
而在实际情况下，即使某特征值的数据分布不是正态分布，但如果数据量够大，也可以使用正态分布的结论。  
既然特征值的数据分布可以假设为正态分布，那么，为了使得多个特征值更为均衡的关联分析，就可以将所有特征值均转化为标准正态分布，即将特征值减去均值进行中心化，再除以标准差进行缩放。

我们假设样本集 $$ S = \{\vec{x}_1,\vec{x}_2,...,\vec{x}_N\} $$，其中样本 $$ \vec{x}_i = (x_{i,1},x_{i,2},...,x_{i,k}) $$ 是表示 $$ k $$ 个特征项的 $$ k $$ 维向量。   
那么，单个特征项的标准差标准化即为：

$$
x_{i,j}' = \frac{x_{i,j} - mean(x_{*,j})}{\sigma_j} 
$$

需要注意，标准差标准化 是基于特征向量的某单个特征项进行标准化，而不考虑各个特征项之间的关系。

在 sklearn 中，通过函数 [sklearn.preprocessing.scale](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html) 可直接进行标准差标准化，scale 后的数据具有零均值以及标准方差：

{%ace edit=false, lang='java'%}
from sklearn import preprocessing
import numpy as np

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
X_scaled = preprocessing.scale(X_train)
print(X_scaled)
// [[ 0.         -1.22474487  1.33630621]
//  [ 1.22474487  0.         -0.26726124]
//  [-1.22474487  1.22474487 -1.06904497]]
print(X_scaled.mean(axis=0))
// [0. 0. 0.]
print(X_scaled.std(axis=0))
// [1. 1. 1.]
{%endace%}

为记录均值及标准差供以后使用，以及在 Pipeline 中使用，sklearn 另提供了一个处理类 [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)：

{%ace edit=false, lang='java'%}
from sklearn import preprocessing
import numpy as np

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
scaler = preprocessing.StandardScaler().fit(X_train)
print(scaler.mean_)
// [1.         0.         0.33333333]
print(scaler.scale_)
// [0.81649658 0.81649658 1.24721913]
print(scaler.transform(X_train))
// [[ 0.         -1.22474487  1.33630621]
//  [ 1.22474487  0.         -0.26726124]
//  [-1.22474487  1.22474487 -1.06904497]]
X_test = [[-1., 1., 0.]]
print(scaler.transform(X_test))
// [[-2.44948974  1.22474487 -0.26726124]]
{%endace%}

## 2. 范围缩放

范围缩放(Range Scaling) 是指将特征缩放至特定范围内（通常是 0~1 之间）。

仍假设样本集 $$ S = \{\vec{x}_1,\vec{x}_2,...,\vec{x}_N\} $$，其中样本 $$ \vec{x}_i = (x_{i,1},x_{i,2},...,x_{i,k}) $$ 是表示 $$ k $$ 个特征项的 $$ k $$ 维向量。   
那么，单个特征项的缩放为：

$$
x_{i,j}' = \frac{x_{i,j} - \min(x_{*,j})}{\max(x_{*,j}) - \min(x_{*,j})}
$$

在 sklearn 中，通过类 [sklearn.preprocessing.MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) 实现范围缩放：

{%ace edit=false, lang='java'%}
from sklearn import preprocessing
import numpy as np

X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
print(X_train_minmax)
// [[0.5        0.         1.        ]
//  [1.         0.5        0.33333333]
//  [0.         1.         0.        ]]
print(min_max_scaler.scale_)
// [0.5        0.5        0.33333333]
print(min_max_scaler.min_)
// [0.         0.5        0.33333333]
{%endace%}

类 [sklearn.preprocessing.MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html) 通过除以单特征项的最大绝对值，将范围缩放到 [-1,1]。

与函数 [sklearn.preprocessing.scale](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html) 类似，我们也可以直接通过函数 [sklearn.preprocessing.minmax_scale](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html) 和  [sklearn.preprocessing.maxabs_scale](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.maxabs_scale.html) 实现以上功能。

## 3. 离群值缩放

离群值（Outliers）也叫异常值，是指其数值明显偏离其余的特征值。对于离群值的检测，如果数据分布满足正态分布，则可以通过 $$ 3\sigma $$ 原则判别。而对于不确定分布的数据，则通过 [箱型图](https://blog.csdn.net/clairliu/article/details/79217546) 来检测。

箱型图 检测，是将数据不属于 $$[Q_L - 1.5 IQR,\quad Q_U + 1.5 IQR]$$ 范围的数据视作离群值。
> $$Q_L$$：下四分位，全部数据中有1/4的数据比他小；
> $$Q_U$$：上四分位，全部数据中有1/4的数据比他大；
> $$IQR$$：四分位间距，是 $$Q_U$$ 和 $$Q_L$$ 的差，其间包含了观察值的一半；

离群值缩放（Scaling Data With Outliers），是指剔除 离群值 后的缩放。sklearn 中，通过函数 [sklearn.preprocessing.robust_scale](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.robust_scale.html) 和类 [sklearn.preprocessing.RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) 实现。

{%ace edit=false, lang='java'%}
from sklearn import preprocessing

X = [[ 1., -2.,  2.],
     [ -2.,  1.,  3.],
     [ 4.,  1., -2.]]
transformer = preprocessing.RobustScaler().fit(X)
print(transformer.center_)
// [1. 1. 2.]
print(transformer.scale_)
// [3.  1.5 2.5]
print(transformer.transform(X))
// [[ 0.  -2.   0. ]
//  [-1.   0.   0.4]
//  [ 1.   0.  -1.6]]
{%endace%}

## 4. 稀疏矩阵缩放

稀疏矩阵中，存在大量零元素，如果进行中心化操作，会使得大量零元素称为非零元素，因此，稀疏矩阵不能进行中心化处理。细化到代码中，需要设置参数 $$ with\_mean=False $$。

