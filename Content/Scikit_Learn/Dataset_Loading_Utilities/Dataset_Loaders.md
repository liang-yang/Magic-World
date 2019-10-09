<!-- toc -->

# Dataset Loaders(Toy Datasets)

---

see [Dataset Loaders](https://scikit-learn.org/stable/datasets/index.html#toy-datasets)

Toy Datasets 一般数据量较小，可以快速验证算法模型。

## 1. Regression Dataset

title | samples | features | description | function
:-:|:-:|:-:|:-:|:-:
波士顿房屋价格 (Boston House Prices) | 506 | 13 | 基于地段、人口密度等特征值，预测波士顿房屋价格 | [sklearn.datasets.load_boston](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html)
糖尿病数据集 (Diabetes) | 442 | 10 | 基于年龄、性别等特征值，预测糖尿病征值 | [sklearn.datasets.load_diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
体能训练数据集 (Linnerrud) | 20 | 3 | 与其他数据集不同的是，此数据集的 y 不再是单列数据，而是 3 列数据，可以做 多项回归 | [sklearn.datasets.load_linnerud](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_linnerud.html)

{%ace edit=true, lang='java'%}
from sklearn.datasets import load_boston

boston = load_boston()
print(boston.data[0:5])
print(boston.target[0:5])
print(boston.feature_names)
print(boston.DESCR)
{%endace%}

## 2. Classification Dataset

title | classes | samples per class | samples | features | description | function
:-:|:-:|:-:|:-:|:-:|:-:|:-:
鸢尾花数据集（Iris Plants） | 3 | 50 | 150 | 4 | 基于花萼、花瓣长、宽等特征值，预测鸢尾花的分类。三个分类中，一个分类与另两个分类线性可分，剩余两个分类线性不可分。 | [sklearn.datasets.load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
视觉识别手写数字数据集（Handwritten Digits）| 10 | ~180 | 1797 | 64 | 图像数据通过8*8的像素矩阵表示，共有64个像素特征值，其中每个元素都是0到16之间的整数。1个目标分类，用来标记图像样本代表的数字，范围是0~9。 | [sklearn.datasets.load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
酒类识别数据集（Wine Recognition）| 3 | [59,71,48] | 178 | 13 | Wine数据集是一个多分类数据集 | [sklearn.datasets.load_wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)
乳腺癌数据集（Breast Cancer）| 2 | [212,357] | 569 | 30 | 乳腺癌数据集是一个二分类数据集 | [sklearn.datasets.load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

{%ace edit=true, lang='java'%}
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.data[0:5])
print(iris.target[0:5])
print(iris.feature_names)
print(iris.DESCR)
{%endace%}
