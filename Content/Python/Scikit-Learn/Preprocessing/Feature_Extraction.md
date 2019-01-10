<!-- toc -->

# Feature Extraction

---

[Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html) 主要是从 text 和 image 中提取特征。

## 1. Loading features from dicts

Python 原生的 dict 数据结构，虽然不太快，但具有简单、稀疏、可读等特点，常用作数据存储。但是，在 scikit-learn 使用时，我们需要通过 类 [DictVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html) 将 dict 转化为 numpy.array 或 SciPy.CSR 格式。

[DictVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html) 将无序的离散分类特征（字符串）转化为 one-of-K（one-hot）编码，数值型特征保持不变。


Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
dtype | callable | numpy.float64 | 转换后array或矩阵的元素类型 | -
separator | string | '=' | one-hot 编码时使用的属性分隔符 | -
sparse | boolean | True | 是否转化为 稀疏矩阵 | -
sort | boolean | True | 转化时 特征名称 是否排序 | -


Attributes | Data-Type | Comment | Note
:-:|:-:|:-:|:-:
vocabulary_ | dict | 特征名称的字典，value为特征排序 | -
feature_names_ | list | 特征名称的数组 | -


Methods | Parameters | Returns | Comment | Note
:-:|:-:|:-:|:-:|:-:
fit | X（特征的字典）| - | 训练特征名称 | -
transform | X（特征的字典）| Xa（转化后的特征向量）| 将特征转化为array或CSR | -
fit_transform | X（特征的字典）| Xa（转化后的特征向量）| 训练并转化特征向量 | -
get_feature_names | - | 特征名称的数组 | - | -
inverse_transform | X（样本的特征矩阵）| D（样本的特征映射）| 将数组或稀疏矩阵X转换回特征映射 | -
restrict | support | - | 对支持使用特征选择的模型进行特征限制，例如只选择前几个特征 | -


{%ace edit=true, lang='python'%}
from sklearn.feature_extraction import DictVectorizer

measurements = [{'city':'Beijing','country':'CN','temperature':33.},{'city':'London','country':'UK','temperature':12.},{'city':'San Fransisco','country':'USA','temperature':18.}]
vec = DictVectorizer()
measurements_vec = vec.fit_transform(measurements)
print(vec.feature_names_)
print(vec.vocabulary_)

print(measurements_vec)
print(measurements_vec.toarray())
{%endace%}

> 通过示例可以看出，多个离散分类特征（字符串）的 one-hot 编码的特征会加到一起，并不会做笛卡尔积。


## 2. Feature hashing

Feature hashing 是降维的一种手段，将 高维度特征向量 转化为 低维度特征向量，主要通过类 [FeatureHasher](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html) 实现。


Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
n_features | integer | 1048576 | 输出矩阵的特征列数量 | 此值太小较容易引起特征碰撞
input_type | string | 'dict' | 输入的数据类型 | -
dtype | numpy type | np.float64 | 特征值的类型，默认为浮点型 | -
alternate_sign | boolean | True | 不同特征hash到同一特征时是否交替变化正负符号，以抵消hash碰撞的影响 | -


Methods | Parameters | Returns | Comment | Note
:-:|:-:|:-:|:-:|:-:
transform | raw_X（将被hash的特征集合）| X（hash后的系数矩阵）| 特征hash | -


{%ace edit=true, lang='python'%}
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import FeatureHasher

measurements = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]

vec = DictVectorizer()
measurements_vec = vec.fit_transform(measurements)
print(measurements_vec)
print(measurements_vec.toarray())

hash = FeatureHasher(n_features=10)
measurements_hash = hash.transform(measurements)
print(measurements_hash)
print(measurements_hash.toarray())
{%endace%}


## 3. Text feature extraction







## 4. Image feature extraction
















