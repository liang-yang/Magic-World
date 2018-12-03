<!-- toc -->

# Datasets

---

## 1. Classification

[sklearn.datasets.make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification)

make_classification 可以随机生成一个N分类的数据集，其生成分类数据的方式是借助于超立方体（hypercube），步骤如下：

1. 根据参数 $$n\_informative$$，超立方体可具有的顶点最大数量为 $$2^{n\_informative}$$，这也是能生成的 $$cluster$$ 的最大数量；
2. 根据参数 $$ n\_classes $$ 和 $$ n\_clusters\_per\_class $$，$$ n\_classes \times n\_clusters\_per\_class$$ 为真正生成的 $$cluster$$ 数量；
3. 根据参数 $$ class\_sep $$，确定各个 $$cluster$$ 之间的距离，可以认为 $$ 2 \times class\_sep $$ 是超立方体的边长；
4. 根据参数 $$ n\_classes $$ 和 $$ n\_clusters\_per\_class $$，把 $$cluster$$ 随机分散到各个 $$class$$，再通过正态分布随机生成样本点；

### 1.1 Parameters

Parameter | Default | Comment | Note
:-:|:-:|:-:|:-:
n_samples | 100 | 样本的数量 | -
n_features | 20 | 总特征的数量，包含 n_informative、n_redundant、n_repeated，剩余部分为随机噪音 | -
n_informative | 2 | 有效特征的数量 | -
n_redundant | 2 | 多余特征的数量 | 由 有效特征 随机线性组合生成
n_repeated | 0 | 重复特征的数量 | 随机从 有效特征 和 多余特征 中选取
n_classes | 2 | 分类的数量 | -
n_clusters_per_class | 2 | 每分类中集群的数量 | -
weights | None | 每个分类占样本数量的比例 | -
flip_y | 0.01 | 随机交换各分类的样本的比例 | 目的是产生一定的噪音
class_sep | 1.0 | 各个 cluster 之间的距离，可认为超立方体的边长 | 较大的值会使得各 cluster 分隔的更远
hypercube | True | 集群点是否放在超立方体的各个顶点 | 最好为True，各集群能分的更开
shift | 0.0 | 位移距离，feature 值的变化范围为 [shift-class_sep, shift+class_sep] | -
scale | 1.0 | 伸缩比例，feature值的变化范围为 [shift-class\_sep\*scale, shift+class\_sep\*scale] | 伸缩在位移之后
shuffle | True | 打乱 样本集 和 特征值 的顺序 | -
random_state | None | 随机种子，使得实验可重复 | -

### 1.2. Returns

Parameter | Comment | Tips
:-:|:-:|:-:
X | 生成的样本集 | -
y | 样本的分类标签 | -




