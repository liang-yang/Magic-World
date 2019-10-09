<!-- toc -->

# Non Linear Transformation

---

在 Standardization 处理中，标准差标准化、范围缩放 等均是线性变换（减去某值，除以某值）。当存在 离群值 时，线性变换虽然可以减小其数值，但数值之间的差距其实是没有改变的。简单的说，线性变换后，离群值依然是离群值。因此，引入了 **非线性变换**。

## 1. 均匀分布变换

非线性变换的一种场景，是通过一定的映射关系，将 非均匀分布 变换为 均匀分布。 分位数变换 就是其中一种方法。

我们知道，特征值可能是非均匀分布的，也会存在离群值的可能。但是，排序肯定是均匀分布的。无论某数值多么离群，大小排序总是顺序的。又由于排序大小与样本量有关，我们吸取 范围缩放 的思路，将排序转化为百分比，想办法将取值限定在 [0, 1] 的范围内。最终，得到了 分位数变换。

sklearn 中，通过类 [sklearn.preprocessing.QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html) 和函数 [sklearn.preprocessing.quantile_transform](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.quantile_transform.html) 实现分位数变换。在具体实现中，主要是通过 numpy 的 [percentile](http://www.360doc.com/content/15/0718/17/26365336_485725251.shtml) 函数完成。下面结合代码对算法进行分析。

{%ace edit=false, lang='java'%}
import numpy as np
from sklearn import preprocessing

X_train = np.array([0,1,2,3,4,5,6,7,8,9,100]).reshape((11, 1))
X_test = np.array([2.5,40]).reshape((2, 1))
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
print(X_train)
// [[  0]
//  [  1]
//  [  2]
//  [  3]
//  [  4]
//  [  5]
//  [  6]
//  [  7]
//  [  8]
//  [  9]
//  [100]]
print(quantile_transformer.fit_transform(X_train))
// [[0. ]
//  [0.1]
//  [0.2]
//  [0.3]
//  [0.4]
//  [0.5]
//  [0.6]
//  [0.7]
//  [0.8]
//  [0.9]
//  [1. ]]
print(quantile_transformer.transform(X_test))
// [[0.25      ]
//  [0.93406593]]
{%endace%}

> 可以看出：
> 1. 训练数据经过分位数变换后，特征向量变化为：$$[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]$$。说明离群值 100 对其余点的分位数并没有太大影响；
> 2. 测试数据中 2.5 的分位数为 0.25，也说明离群值对其分位值没有影响；
> 3. 测试数据中 40 的分位数为 0.9340，说明在 9 ~ 100 的范围内，仅有10%的分布空间。也就是说，将数值中 9 ~ 100 这么大的区间映射为了分位数中 0.9 ~ 1 这么小的区间；

综上，分位数变换 可以理解为对所有已排序的相邻数据赋予同等的区间。离群值肯定较正常值更为稀疏，以此实现非线性的变换，进而规避离群值的影响。

另外，也可以从概率分布的角度来理解。上例中，小于9的概率达到90%，小于40的概率为93.4%，那么 9~40 的概率为3.40%

## 2. 高斯分布变换

在许多建模场景中，需要数据满足正态分布，因此我们有需求将任意连续分布变换为高斯分布。理论上，这一点也是可实现的。

我们假设 $$x \sim F$$ 为一个连续分布，根据概率分布的定义，$$F(x) \in [0,1]$$。我们定义 $$ y = \Phi^{-1}(F(x)) $$，其中 $$\Phi^{-1}$$ 是标准正态分布函数的逆函数。那么经过这次逆运算，$$F(x)$$ 映射的 $$y \sim N(0, 1)$$，即经过变换后满足了标准正态分布。

因此，问题的关键是我们需要知道变量 $$x$$ 的分布函数 $$F(x)$$。实际上，我们可以利用前述的 分位数变换，将变量 $$x$$ 变换为分位数 $$u$$，此时分位数 $$u$$ 就近似于概率分布函数 $$F(x)$$。

完整的转换流程为：

$$
x -\text{分位数变换} \to u -\text{正态分布逆函数} \to y
$$

最终将原特征值变换为了标准正态分布。

> [参考](https://www.zhihu.com/question/311540570/answer/595153078)

这种高斯分布变换的思路，可通过在类 [sklearn.preprocessing.QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html) 中设置参数 output_distribution='normal' 来实现。

另外，在 sklearn 中可通过类 [sklearn.preprocessing.PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html) 使用 **幂变换** 进行高斯分布变换。PowerTransformer 支持 Box-Cox转换 和 Yeo-Johnson转换。通过最大似然估计出稳定方差和最小偏度的最优参数。Box-Cox要求输入数据严格为正，而Yeo-Johnson同时支持正或负数据。默认情况下，对转换后的数据应用零均值、单位方差归一化。


{%ace edit=false, lang='java'%}
import numpy as np
from sklearn import preprocessing

pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
X_lognormal = np.random.RandomState(616).lognormal(size=(3, 3))
print(X_lognormal)
// [[1.28331718 1.18092228 0.84160269]
//  [0.94293279 1.60960836 0.3879099 ]
//  [1.35235668 0.21715673 1.09977091]]
print(pt.fit_transform(X_lognormal))
// [[ 0.49024349  0.17881995 -0.1563781 ]
//  [-0.05102892  0.58863195 -0.57612415]
//  [ 0.69420008 -0.84857822  0.10051454]]
{%endace%}
