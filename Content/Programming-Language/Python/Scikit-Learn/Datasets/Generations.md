<!-- toc -->

# Generations

---

## 1. classification and clustering

### 1.1. make-blobs

[sklearn.datasets.make_blobs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)

Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
n_samples | int, array | 100 | 如果为int，表示样本总数，会在各集群中平均分配；如果为array，表示各个集群的样本数量 | -
n_features | int | 2 | 每样本的特征数量 | -
centers | int, array[n_centers,n_features] | None | - | -
cluster_std | float, sequence of floats | 1.0 | 集群的标准差 | -
center_box | pair of floats (min,max) | (-10.0,10.0) | 每个集群均值的边界范围 | -



### 1.2. make-classification


### 1.3. make_gaussian_quantiles


### 1.4. make_hastie_10_2


### 1.5. make_circles 


### 1.6. make_moons


### 1.7. make_multilabel_classification


### 1.8. make_biclusters


### 1.9. make_checkerboard





## 1. classification

[sklearn.datasets.make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)

make_classification 可以随机生成一个N分类的数据集，其生成分类数据的方式是借助于超立方体（hypercube），步骤如下：

1. 根据参数 $$n\_informative$$，超立方体可具有的顶点最大数量为 $$2^{n\_informative}$$，这也是能生成的 $$cluster$$ 的最大数量；
2. 根据参数 $$ n\_classes $$ 和 $$ n\_clusters\_per\_class $$，$$ n\_classes \times n\_clusters\_per\_class$$ 为真正生成的 $$cluster$$ 数量；
3. 根据参数 $$ class\_sep $$，确定各个 $$cluster$$ 之间的距离，可以认为 $$ 2 \times class\_sep $$ 是超立方体的边长；
4. 根据参数 $$ n\_classes $$ 和 $$ n\_clusters\_per\_class $$，把 $$cluster$$ 随机分散到各个 $$class$$，再通过正态分布随机生成样本点；

### 1.1. Parameters

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

### 1.2. Returns

Returns | Data-Type | Comment | Note
:-:|:-:|:-:|:-:
X | array[n_samples,n_features] | 生成的样本集 | -
y | array[n_samples] | 样本的分类标签 | -

