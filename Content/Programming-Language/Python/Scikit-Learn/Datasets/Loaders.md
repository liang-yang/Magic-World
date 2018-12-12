<!-- toc -->

# Loaders(Toy datasets)

---

## 1. Boston house prices

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

## 2. Iris plants

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

## 3. Diabetes

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

## 4. Optical recognition of handwritten digits dataset

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

## 5. Linnerrud
 
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

## 6. Wine recognition

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

## 7. Breast cancer

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
