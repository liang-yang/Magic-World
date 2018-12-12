<!-- toc -->

# Datasets

---

[sklearn.datasets](https://scikit-learn.org/stable/datasets/index.html)

Datasets 提供 三 种数据集接口：Loaders、Fetchers 和 Generations。

- **Loaders**：加载 小数据量数据集，也称 Toy datasets；
- **Fetchers**：下载 并 加载 大数据量数据集，也称 Real world datasets；
- **Generations**：根据输入参数人为控制生成数据集；

他们都会返回：

- X: array[n_samples * n_features]
- y: array[n_samples]

对于 Loaders 和 Fetchers，还可以通过 **DESCR** 获取 特征列表。

## 1. Loaders(Toy datasets)

### 1.1. Boston house prices

[sklearn.datasets.load_boston](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html)

波士顿房屋价格的数据集，常用于 regression

samples | features 
:-:|:-:
506 | 13 

{%ace edit=true, lang='python'%}
from sklearn.datasets import load_boston

boston = load_boston()
print(boston.data[0:5])
print(boston.target[0:5])
print(boston.feature_names)
print(boston.DESCR)
{%endace%}

### 1.2. Iris plants

[sklearn.datasets.load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)

鸢尾花数据集，常用于 classification

classes | samples per class | samples | features |  
:-:|:-:|:-:|:-:
3 | 50 | 150 | 4 

{%ace edit=true, lang='python'%}
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.data[0:5])
print(iris.target[0:5])
print(iris.feature_names)
print(iris.DESCR)
{%endace%}

### 1.3. Diabetes

[sklearn.datasets.load_diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)

糖尿病数据集，常用于 regression

samples | features 
:-:|:-:
442 | 10 

{%ace edit=true, lang='python'%}
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
print(diabetes.data[0:5])
print(diabetes.target[0:5])
print(diabetes.feature_names)
print(diabetes.DESCR)
{%endace%}

### 1.4. Optical recognition of handwritten digits dataset

[sklearn.datasets.load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)

视觉识别手写数字的数据集，常用于 classification

classes | samples per class | samples | features |  
:-:|:-:|:-:|:-:
10 | ~180 | 1797 | 64 

{%ace edit=true, lang='python'%}
from sklearn.datasets import load_digits

digits = load_digits()
print(digits.data[0:5])
print(digits.target[0:5])
print(digits.DESCR)
{%endace%}

### 1.5. Linnerrud
 
[sklearn.datasets.load_linnerud](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_linnerud.html)

体能训练数据集。与其他数据集不同的是，此数据集的 y 不再是单列数据，而是 3 列数据，可以做 多项回归。

samples | features of X | features of y
:-:|:-:|:-:
20 | 3 | 3

{%ace edit=true, lang='python'%}
from sklearn.datasets import load_linnerud

linnerud = load_linnerud()
print(linnerud.data)
print(linnerud.target)
print(linnerud.feature_names)
print(linnerud.target_names)
{%endace%}

### 1.6. Wine recognition

[sklearn.datasets.load_wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)

酒类识别数据集，常用于 classification

classes | samples per class | samples | features |  
:-:|:-:|:-:|:-:
3 | [59,71,48] | 178 | 13 

{%ace edit=true, lang='python'%}
from sklearn.datasets import load_wine

wine = load_wine()
print(wine.data[0:5])
print(wine.target[0:5])
print(wine.DESCR)
{%endace%}

### 1.7. Breast cancer

[sklearn.datasets.load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

乳腺癌数据集，常用于 binary classification

classes | samples per class | samples | features |  
:-:|:-:|:-:|:-:
2 | 212(M),357(B) | 569 | 30 

{%ace edit=true, lang='python'%}
from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()
print(breast_cancer.data[0:5])
print(breast_cancer.target[0:5])
print(breast_cancer.DESCR)
{%endace%}

## 2. Fetchers(Real world datasets)

### 2.1. The Olivetti faces dataset

[sklearn.datasets.fetch_olivetti_faces](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html)

人脸识别的数据集，可以认为有 400 张图片，每张图片的像素均为 64*64=4096. 

classes | samples | features |  
:-:|:-:|:-:
40 | 400 | 4096 

{%ace edit=true, lang='python'%}
from sklearn.datasets import fetch_olivetti_faces

olivetti_faces = fetch_olivetti_faces()
print(olivetti_faces.data[0:1])
print(olivetti_faces.images[0:1])
print(olivetti_faces.target[0:1])
print(olivetti_faces.DESCR)
{%endace%}

### 2.2. The 20 newsgroups text dataset

新闻数据集，将 18846 条新闻划分为 20 类。此数据集可通过 fetch_20newsgroups 和 fetch_20newsgroups_vectorized 两个函数获取，前者返回文本，后者返回特征向量。

#### 2.2.1. fetch_20newsgroups

[sklearn.datasets.fetch_20newsgroups](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html)



#### 2.2.2. fetch_20newsgroups_vectorized

[sklearn.datasets.fetch_20newsgroups_vectorized](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups_vectorized.html)





## 3. Generations



### 3.1. make_classification

[sklearn.datasets.make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification)

make_classification 可以随机生成一个N分类的数据集，其生成分类数据的方式是借助于超立方体（hypercube），步骤如下：

1. 根据参数 $$n\_informative$$，超立方体可具有的顶点最大数量为 $$2^{n\_informative}$$，这也是能生成的 $$cluster$$ 的最大数量；
2. 根据参数 $$ n\_classes $$ 和 $$ n\_clusters\_per\_class $$，$$ n\_classes \times n\_clusters\_per\_class$$ 为真正生成的 $$cluster$$ 数量；
3. 根据参数 $$ class\_sep $$，确定各个 $$cluster$$ 之间的距离，可以认为 $$ 2 \times class\_sep $$ 是超立方体的边长；
4. 根据参数 $$ n\_classes $$ 和 $$ n\_clusters\_per\_class $$，把 $$cluster$$ 随机分散到各个 $$class$$，再通过正态分布随机生成样本点；

#### 3.1.1. Parameters

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
shuffle | boolean | True | 打乱 样本集 和 特征值 的顺序 | -
random_state | int, RandomState, None | None | 随机种子，使得实验可重复 | -

#### 3.1.2. Returns

Returns | Data-Type | Comment | Note
:-:|:-:|:-:|:-:
X | array[n_samples,n_features] | 生成的样本集 | -
y | array[n_samples] | 样本的分类标签 | -


