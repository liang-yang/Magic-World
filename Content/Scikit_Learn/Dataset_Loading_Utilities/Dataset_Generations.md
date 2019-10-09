<!-- toc -->

# Dataset Generations

---

see [Dataset Generations](https://scikit-learn.org/stable/datasets/index.html#generated-datasets)

Dataset Generations 是根据输入参数人为控制统计属性生成数据集。

## 1. Classification And Clustering

分类 和 聚类 在数据集层面都是一样的，都是将整个数据集分离为多个数据集群，因此这里将两者归为一类。

### 1.1. make-blobs

[sklearn.datasets.make_blobs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)

make_blobs 可通过制定 均值 和 标准差 来生成数据集。

Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
n_samples | int, array | 100 | 如果为int，表示样本总数，会在各集群中平均分配；如果为array，表示各个集群的样本数量 | -
n_features | int | 2 | 每样本的特征数量 | -
centers | int, array[n_centers,n_features] | None | 集群数量。如果n_samples为int，且centers为None，则集群数量为3。如果n_samples为array，则centers需为array[n_centers,n_features] | 此参数可以认为是此函数最重要的参数，实际上它是指定了各集群的各特征的均值。如果为整数，则会通过随机数生成
cluster_std | float, sequence of floats | 1.0 | 集群的标准差 | -
center_box | pair of floats (min,max) | (-10.0,10.0) | 随机生成的集群均值的边界范围 | -

{%ace edit=true, lang='java'%}
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=100, centers=[[1, 1], [-10, 0]], random_state=0)

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()
{%endace%}

### 1.2. make-classification

[sklearn.datasets.make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)

make_classification 最大的特点是可以加入 噪音 影响。

make_classification 生成分类数据的方式是借助于超立方体（hypercube），步骤如下：

1. 根据参数 $$n\_informative$$，超立方体可具有的顶点最大数量为 $$2^{n\_informative}$$，这也是能生成的 $$cluster$$ 的最大数量；
2. 根据参数 $$ n\_classes $$ 和 $$ n\_clusters\_per\_class $$，$$ n\_classes \times n\_clusters\_per\_class$$ 为真正生成的 $$cluster$$ 数量；
3. 根据参数 $$ class\_sep $$，确定各个 $$cluster$$ 之间的距离，可以认为 $$ 2 \times class\_sep $$ 是超立方体的边长；
4. 根据参数 $$ n\_classes $$ 和 $$ n\_clusters\_per\_class $$，把 $$cluster$$ 随机分散到各个 $$class$$，再通过正态分布随机生成样本点；

Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
n_samples | int | 100 | 样本的数量 | -
n_features | int | 20 | 总特征的数量，包含 n_informative、n_redundant、n_repeated，剩余部分为随机噪音 | -
n_informative | int | 2 | 有效特征的数量 | -
n_redundant | int | 2 | 多余特征的数量 | 由 有效特征 随机线性组合生成
n_repeated | int | 0 | 重复特征的数量 | 随机从 有效特征 和 多余特征 中选取
n_classes | int | 2 | 分类的数量 | -
n_clusters_per_class | int | 2 | 每分类中集群的数量 | -
weights | list of floats, None | None | 每个分类占样本数量的比例 | -
flip_y | float | 0.01 | 随机交换各分类的样本的比例 | 目的是产生一定的噪音
class_sep | float | 1.0 | 各个 cluster 之间的距离，可认为超立方体的边长 | 较大的值会使得各 cluster 分隔的更远
hypercube | boolean | True | 集群点是否放在超立方体的各个顶点 | 最好为True，各集群能分的更开
shift | float, array[n_features], None | 0.0 | 位移距离，feature 值的变化范围为 [shift-class_sep, shift+class_sep] | -
scale | float, array[n_features], None | 1.0 | 伸缩比例，feature值的变化范围为 [shift-class\_sep\*scale, shift+class\_sep\*scale] | 伸缩在位移之后

Returns | Data-Type | Comment | Note
:-:|:-:|:-:|:-:
X | array[n_samples,n_features] | 生成的样本集 | -
y | array[n_samples] | 样本的分类标签 | -

{%ace edit=true, lang='java'%}
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=200, n_features=2, n_informative = 2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2, class_sep=5, random_state=0)

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()
{%endace%}

### 1.3. make-gaussian-quantiles

[sklearn.datasets.make_gaussian_quantiles](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_gaussian_quantiles.html)

make_gaussian_quantiles 是生成环形的分类，步骤如下：

1. 以 mean 和 cov 作为正态分布的参数，对各个特征独立的随机生成特征值；
2. 基于每个样本点与 mean 点的距离对样本点进行排序；
3. 根据 n_samples 和 n_classes 确定每个分类的样本点数量，然后根据距离排序顺序分配；

Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
mean | array[n_features] | None | 各特征的均值，如果为None，则均为0 | -
cov | float | 1.0 | 各个特征的方差 | -
n_samples | int | 100 | 总样本数量 | -
n_features | int | 2 | 特征数量 | -
n_classes | int | 3 | 分类的数量 | -

{%ace edit=true, lang='java'%}
from sklearn.datasets import make_gaussian_quantiles
import matplotlib.pyplot as plt

X, y = make_gaussian_quantiles(n_samples=100, n_features=2, random_state=0)

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()
{%endace%}

### 1.4. make-hastie-10-2

[sklearn.datasets.make_hastie_10_2](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_hastie_10_2.html)

make_hastie_10_2 基于正态分布，独立生成10个特征的数据，并根据如下公式将其分为二类：

$$
y[i] = \begin{cases}
   1 &\text{if } np.sum(X[i] ** 2) > 9.34 \\
   -1 &\text{if } np.sum(X[i] ** 2) \leqslant 9.34
\end{cases}
$$

{%ace edit=true, lang='java'%}
from sklearn.datasets import make_hastie_10_2
import matplotlib.pyplot as plt

X, y = make_hastie_10_2(n_samples=100, random_state=0)

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()
{%endace%}

### 1.5. make-circles 

[sklearn.datasets.make_circles](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html)

make_circles 生成内外两层圆圈，步骤如下：

1. 根据参数 $$ 2 \pi$$ 等分为 n_samples 份，得到每个样本的角度；
2. 令 $$ x=\cos,y=\sin $$，将角度转化为坐标，此为外圈样本的坐标；
3. 将外圈坐标乘以 factor，得到内圈样本的坐标；
4. 以 noise 为标准差形成零均值的正态分布，作为噪音加到原有X上；

Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
noise | double | None | 噪音的标准差 | -
factor | double[0,1] | 0.8 | 内圈占外圈的比例因子 | -

{%ace edit=true, lang='java'%}
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

X, y = make_circles(n_samples=100, noise=0.01, random_state=0)

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()
{%endace%}

### 1.6. make-moons

[sklearn.datasets.make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)

make_moons 生成两个交错的半圆，步骤基本与 make_circles 类似，只是将圆换成半圆，并且进行了一定的位移。

{%ace edit=true, lang='java'%}
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=100, noise=0.01, random_state=0)

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()
{%endace%}

### 1.7. make-multilabel-classification



### 1.8. make-biclusters



### 1.9. make-checkerboard


## 2. Regression

### 2.1. make-regression

[sklearn.datasets.make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)

Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
n_samples | int | 100 | 总样本数量 | -
n_features | int | 100 | 总特征数量 | -
n_informative | int | 10 | 有用的特征数量 | -
n_targets | int | 1 | 输出的回归值的维度数量，默认为1，即标量 | -
bias | float | 0.0 | 偏差项 | -
effective_rank | int | None |  | -
tail_strength | float[0,1] | 0.5 |  | -
noise | float | 0.0 | 噪音的标准差 | -
coef | boolean | False | 是否返回系数 | -

Returns | Data-Type | Comment | Note
:-:|:-:|:-:|:-:
X | array[n_samples,n_features] |  | -
y | array[n_samples,n_targets] |  | -
coef | array[n_features,n_targets] |  | -
 
{%ace edit=true, lang='java'%}

{%endace%}

### 2.2. make-sparse-uncorrelated





### 2.3. make_friedman1





### 2.4. make_friedman2





### 2.5. make_friedman3




## 3. Manifold

### 3.1. make-s-curve


### 3.2. make-swiss-roll



## 4. Decomposition

### 4.1. make-low-rank-matrix

https://zhidao.baidu.com/question/172378440.html



### 4.2. make-sparse-coded-signal




### 4.3. make-spd-matrix




### 4.4. make-sparse-spd-matrix














