<!-- toc -->

# Feature Extraction

---

[Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html) 主要是从 text 和 image 中提取特征。

## 1. Loading features from dicts

Python 原生的 dict 数据结构，虽然不太快，但具有简单、稀疏、可读等特点，常用作数据存储。在 scikit-learn 使用时，我们需要通过 类 [DictVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html) 将 dict 转化为 numpy.array 或 CSR 格式。

需要注意，[DictVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html) 



Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
dtype |  | numpy.float64 |  | -
separator |  | '=' |  | -
sparse |  | True |  | -
sort |  | True |  | -


Attributes | Data-Type | Comment | Note
:-:|:-:|:-:|:-:
 |  |  | -
 |  |  | -
 |  |  | -
 |  |  | -
 |  |  | -


## 2. Feature hashing






## 3. Text feature extraction







## 4. Image feature extraction
















