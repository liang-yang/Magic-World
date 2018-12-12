<!-- toc -->

# Fetchers(Real world datasets)

---

## 1. The Olivetti faces dataset

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

## 2. The 20 newsgroups text dataset

新闻数据集，将 18846 条新闻划分为 20 类。此数据集可通过 fetch_20newsgroups 和 fetch_20newsgroups_vectorized 两个函数获取，前者返回文本，后者返回特征向量。

### 2.1. fetch_20newsgroups

[sklearn.datasets.fetch_20newsgroups](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html)

classes | samples | features |  
:-:|:-:|:-:
20 | 18846 | 1 

Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
subset | 'train', 'test', 'all' | 'train' | 加载的数据集 | -
categories | None, collection of string, collection of unicode | None | 分类，即 classes | -
remove | tuple | () | 特征提取时忽略的文本部分，('headers', 'footers', 'quotes')的子集 | 为避免过拟合，特征提取时经常会忽略 标题、页脚、引用 等

Returns | Data-Type | Comment | Note
:-:|:-:|:-:|:-:
bunch.data | list[n_samples] | 样本的新闻文本 | -
bunch.target | list[n_samples] | 样本的新闻种类 | -
bunch.target_names | list[n_classes] | 所有新闻种类的集合 | -

{%ace edit=true, lang='python'%}
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups()
print(newsgroups.data[0:2])
print(newsgroups.target[0:2])
print(newsgroups.target_names)
print(len(newsgroups.target_names))
print(newsgroups.DESCR)
{%endace%}

### 2.2. fetch_20newsgroups_vectorized

[sklearn.datasets.fetch_20newsgroups_vectorized](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups_vectorized.html)

通过 

> sklearn.feature_extraction.text.CountVectorizer
> sklearn.feature_extraction.text.HashingVectorizer
> sklearn.feature_extraction.text.TfidfTransformer
> sklearn.feature_extraction.text.TfidfVectorizer

对 20newsgroups 文本处理后的数据集。

classes | samples | features |  
:-:|:-:|:-:
20 | 18846 | 130107 

{%ace edit=true, lang='python'%}
from sklearn.datasets import fetch_20newsgroups_vectorized

newsgroups = fetch_20newsgroups_vectorized()
print(newsgroups.data[0:2])
print(newsgroups.target[0:2])
print(newsgroups.target_names)
print(len(newsgroups.target_names))
{%endace%}














