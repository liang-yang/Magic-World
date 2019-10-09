<!-- toc -->

# Feature Convert

---

see [特征处理（Feature Processing）](http://breezedeus.github.io/2014/11/15/breezedeus-feature-processing.html)

特征转化，是指将特征值由 便于逻辑理解 的存储格式转化为 便于机器学习 的存储格式。

特征 通常属于如下三类：

- 连续（continuous）特征
- 无序类别（categorical）特征
- 有序类别（ordinal）特征

## 1. 连续特征

对于 连续 特征，其存储格式无论是 逻辑理解 还是 机器学习，使用的都比较一致，因此一般不需要进行什么转化（我们将 归一化 处理归为预处理，不算作特征转化）。

## 2. 无序类别特征

无序特征，例如特征 color，有三种取值：red、green、blue，这就是我们逻辑上对特征进行描述的格式。在常规数据处理中，为了便于存储及处理，我们常常会将 字符串 与 数字 一一映射，即使得 red=1、green=2、blue=3。

> 此种转换方式，在 sklearn 中通过 [sklearn.preprocessing.OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) 实现。

但是，这种转换方式会使得各取值之间可比较大小、甚至可进行运算，例如 1 + 2 = 3 会使得 red + green = blue。但实际上它们之间是不存在这种运算关系的。基于以上原因，在机器学习中，对无序特征的转化一般使用 One-hot（One-of-k）将取值转化为一个数值向量：

color | vector 
:-:|:-:
_red_|$$(1,0,0)$$
_green_|$$(0,1,0)$$
_blue_|$$(0,0,1)$$

这种方法在NLP里用的很多，就是所谓的词向量模型。变换后的向量长度对于词典长度，每个词对应于向量中的一个元素。

> One-hot 转换方式在 sklearn 中通过 [sklearn.preprocessing.OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) 实现。

## 3. 有序类别特征

对于有序特征，取值之间存在一定的大小关系，但不存在运算关系，因此通过如下方式表示：

status | vector 
:-:|:-:
_bad_|$$(1,0,0)$$
_normal_|$$(1,1,0)$$
_good_|$$(1,1,1)$$

这样就利用递进表达了值之间的顺序关系。

## 4. sklearn代码处理

在 scikit-learn 中，如果数据格式为 Python 原生的 dict 数据结构，首先需要将其转化为 numpy.array 或 SciPy.CSR 格式，再进行运算。另外，需要将无序的离散分类特征（字符串）转化为 one-of-K（one-hot）编码。这些，可通过类 [sklearn.feature_extraction.DictVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html) 统一实现。

{%ace edit=true, lang='java'%}
from sklearn.feature_extraction import DictVectorizer

measurements = [{'city':'Beijing','country':'CN','temperature':33.},{'city':'London','country':'UK','temperature':12.},{'city':'San Fransisco','country':'USA','temperature':18.}]
vec = DictVectorizer()
measurements_vec = vec.fit_transform(measurements)
print(vec.feature_names_)
// ['city=Beijing', 'city=London', 'city=San Fransisco', 'country=CN', 'country=UK', 'country=USA', 'temperature']
print(vec.vocabulary_)
// {'city=Beijing': 0, 'country=CN': 3, 'temperature': 6, 'city=London': 1, 'country=UK': 4, 'city=San Fransisco': 2, 'country=USA': 5}

print(measurements_vec)
//   (0, 0)    1.0
//   (0, 3)    1.0
//   (0, 6)    33.0
//   (1, 1)    1.0
//   (1, 4)    1.0
//   (1, 6)    12.0
//   (2, 2)    1.0
//   (2, 5)    1.0
//   (2, 6)    18.0
print(measurements_vec.toarray())
// [[ 1.  0.  0.  1.  0.  0. 33.]
//  [ 0.  1.  0.  0.  1.  0. 12.]
//  [ 0.  0.  1.  0.  0.  1. 18.]]
{%endace%}

通过示例可以看出，多个离散分类特征（字符串）的 one-hot 编码的特征会加到一起。


