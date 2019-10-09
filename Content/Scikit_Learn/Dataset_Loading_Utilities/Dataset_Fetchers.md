<!-- toc -->

# Dataset Fetchers(Real World Datasets)

---

see [Dataset Fetchers](https://scikit-learn.org/stable/datasets/index.html#real-world-datasets)

Real World Datasets 是较之 Toy Datasets 数据量更大的数据集。

## 1. The Olivetti Faces Dataset

see [sklearn.datasets.fetch_olivetti_faces](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html)

人脸识别的数据集，可以认为有 400 张图片，每张图片的像素均为 64*64 = 4096。 

classes | samples | features |  
:-:|:-:|:-:
40 | 400 | 4096 

{%ace edit=true, lang='java'%}
from sklearn.datasets import fetch_olivetti_faces

olivetti_faces = fetch_olivetti_faces()
print(olivetti_faces.data[0:1])
print(olivetti_faces.images[0:1])
print(olivetti_faces.target[0:1])
print(olivetti_faces.DESCR)
{%endace%}

## 2. The 20 Newsgroups Text Dataset

新闻数据集，将 18846 条新闻划分为 20 类。此数据集可通过 fetch_20newsgroups 和 fetch_20newsgroups_vectorized 两个函数获取，前者返回文本，后者返回特征向量。

### 2.1. 20newsgroups

see [sklearn.datasets.fetch_20newsgroups](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html)

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

{%ace edit=true, lang='java'%}
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups()
print(newsgroups.data[0:2])
print(newsgroups.target[0:2])
print(newsgroups.target_names)
print(len(newsgroups.target_names))
print(newsgroups.DESCR)
{%endace%}

### 2.2. 20newsgroups vectorized

see [sklearn.datasets.fetch_20newsgroups_vectorized](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups_vectorized.html)

通过 

> sklearn.feature_extraction.text.CountVectorizer  
> sklearn.feature_extraction.text.HashingVectorizer  
> sklearn.feature_extraction.text.TfidfTransformer  
> sklearn.feature_extraction.text.TfidfVectorizer  

对 20newsgroups 文本处理后的数据集。

classes | samples | features |  
:-:|:-:|:-:
20 | 18846 | 130107 

{%ace edit=true, lang='java'%}
from sklearn.datasets import fetch_20newsgroups_vectorized

newsgroups = fetch_20newsgroups_vectorized()
print(newsgroups.data[0:2])
print(newsgroups.target[0:2])
print(newsgroups.target_names)
print(len(newsgroups.target_names))
{%endace%}

## 3. The Labeled Faces In The Wild Face Recognition

采集的人脸的图片数据集，主要可用作 人脸验证（Face Verification）和 人脸识别（Face Recognition）。

人脸识别模型 Viola-Jones，实现类库 OpenCV。

### 3.1. lfw people

see [sklearn.datasets.fetch_lfw_people](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html)

classes | samples | features |  
:-:|:-:|:-:
5749 | 13233 | 5828 

Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
min_faces_per_person | int | None | 数据集中的人物至少存在多少张图片 | -
color | boolean | False | 是否保留 RGB 3个元素，如果 False 则会转化为 gray 1个元素 | -
slice_ | slice | slice(70,195,None), slice(78,172,None) | 提供一个 slice(height, width) 的长方形以提取图片内容，避免提取过多的背景元素 | -

Returns | Data-Type | Comment | Note
:-:|:-:|:-:|:-:
dataset.data | array(13233,2914) | 每一行对应原始图片 62*47 像素 | 修改 slice_ 参数可能改变返回的 shape 
dataset.images | array(13233,62,47) | 每一行对应原始图片 62*47 像素 | 修改 slice_ 参数可能改变返回的 shape
dataset.target | array(13233) | 每一行对应原始图片所属人物的ID | -
dataset.target_names | array(13233) | 人物名称 | -

{%ace edit=true, lang='java'%}
from sklearn.datasets import fetch_lfw_people

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
print(lfw_people.data[0:2])
print(lfw_people.target[0:2])
print(lfw_people.target_names)
{%endace%}

### 3.2. lfw pairs

see [sklearn.datasets.fetch_lfw_pairs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_pairs.html)

此数据集专用于比较 两张人脸 是否属于同一个人。

classes | samples | features |  
:-:|:-:|:-:
5749 | 13233 | 5828 

Returns | Data-Type | Comment | Note
:-:|:-:|:-:|:-:
dataset.data | array(2200,5828) | 每一行对应两张原始图片 62*47 像素 | shape 依赖于使用的子集
dataset.pairs | array(2200,2,62,47) | 每一行对应两张图片 | -
dataset.target | array(2200) | 两张图片是否对应一个人 | -

{%ace edit=true, lang='java'%}
from sklearn.datasets import fetch_lfw_pairs
lfw_pairs_train = fetch_lfw_pairs(subset='train')

print(list(lfw_pairs_train.target_names))
print(lfw_pairs_train.data[0:2])
print(lfw_pairs_train.pairs[0:2])
print(lfw_pairs_train.target[0:2])
print(lfw_pairs_train.target_names[0:2])
{%endace%}

## 4. Forest Covertypes

see [sklearn.datasets.fetch_covtype](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html)

此数据集提供 581012 个植被（面积为30m*30m），可用以判断每个植被的覆盖类型（种植的植物）。

比较特殊的一点，特征值部分是 boolean 型，部分是 数值 型。

classes | samples | features |  
:-:|:-:|:-:
7 | 581012 | 54

Returns | Data-Type | Comment | Note
:-:|:-:|:-:|:-:
dataset.data | array(581012,54) | 每一行对应包含54个特征值 | -
dataset.target | array(581012) | 7种覆盖类型之一 | -

{%ace edit=true, lang='java'%}
from sklearn.datasets import fetch_covtype
covtype = fetch_covtype()

print(covtype.data[0:2])
print(covtype.target[0:2])
print(covtype.target_names[0:2])
{%endace%}

## 5. RCV1 Dataset

[sklearn.datasets.fetch_rcv1](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_rcv1.html)

路透社语料库，超过 800000 篇人工分类的报道

classes | samples | features |  
:-:|:-:|:-:
103 | 804414 | 47236

Returns | Data-Type | Comment | Note
:-:|:-:|:-:|:-:
dataset.data | csr_array(804414,47236) | 每一行对应包含47236个特征值 | 0.16%的非零值
dataset.target | csr_array(804414,103) | 103个分类标签，仅有0, 1值 | 3.15%的非零值
dataset.target_names | array(103) | 分类的名字 | -
dataset.sample_id | array(804414) | 报道ID | -

{%ace edit=true, lang='java'%}
from sklearn.datasets import fetch_rcv1
rcv1 = fetch_rcv1()

print(rcv1.data[0:2])
print(rcv1.target[0:2])
print(rcv1.target_names[0:2])
{%endace%}

## 6. Kddcup 99 Dataset

[sklearn.datasets.fetch_kddcup99](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html)

此数据集是从一个模拟的美国空军局域网上采集来的9个星期的网络连接数据，分成具有标识的训练数据和未加标识的测试数据。

测试数据和训练数据有着不同的概率分布，测试数据包含了一些未出现在训练数据中的攻击类型，这使得入侵检测更具有现实性。

在训练数据集中包含了1种正常的标识类型normal和22种训练攻击类型。另外有14种攻击仅出现在测试数据集中。

训练数据集中每个连接记录包含了41个固定的特征属性和1个类标识：标识用来表示该条连接记录是正常的，或是某个具体的攻击类型。在41个固定的特征属性中，9个特征属性为离散(symbolic)型，其他均为连续(continuous)型。

classes | samples | features |  
:-:|:-:|:-:
23 | 4898431 | 41

## 7. California Housing Dataset

see [sklearn.datasets.fetch_california_housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

此数据集中包含房屋的8项属性，以预测房屋价格的中值。

samples | features |  
:-:|:-:
20640 | 8

Returns | Data-Type | Comment | Note
:-:|:-:|:-:|:-:
dataset.data | array(20640, 8) | 每一行对应包含8个特征值 | - 
dataset.target | array(20640) | 每一行对应平均楼房价格 | -
dataset.feature_names | array(8) | 特征值的名字 | -

{%ace edit=true, lang='java'%}
from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing()

print(california_housing.data[0:2])
print(california_housing.target[0:2])
print(california_housing.feature_names)
{%endace%}

