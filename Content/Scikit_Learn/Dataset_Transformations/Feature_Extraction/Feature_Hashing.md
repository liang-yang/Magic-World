<!-- toc -->

# Feature Hashing

---

see [Feature Hashing](http://breezedeus.github.io/2014/11/20/breezedeus-feature-hashing.html)

在某些场景，特征向量的维度会很大，这会导致存储量、计算量很大，因此我们需要降维。常见的降维方式有 聚类、PCA 等。但这些方法在特征量和样本量很多的时候本身就计算量很大，所以对大问题也基本无能为力。因此，出现了 **特征哈希** 这种较为直观和简单的降维方法。

**特征哈希** 的目标是把原始的高维特征向量压缩成较低维特征向量，且尽量不损失原始特征的表达能力。

假设哈希前的特征向量为 $$ \vec{x} = (x_1, x_2, ... , x_N) \in R^N $$。我们要把这个原始的 $$ N $$ 维特征向量压缩成 $$ M $$ 维（$$ M < N $$），即 $$ \vec{y} = (y_1, y_2, ... , y_M) \in R^M $$。构造两个哈希函数：

$$
h(n):\{1,2,...,N\} \to \{1,2,...,M\}  
$$

$$
g(n):\{1,2,...,N\} \to \{-1,1\} 
$$

$$ h(n) $$ 和 $$ g(n) $$ 是独立不相关的（这两个函数的入参 $$ n $$ 表示特征向量的第 $$ n $$ 维）。

那么，我们令：

$$
y_i = \sum_{j:h(j)=i} g(j) \cdot x_j
$$

简单理解，就是把 HASH 到一起的两个维度带符号相加。 例如设 $$ N=4 , M=2 $$，令 $$ h(1) = h(3) = 1, h(2) = h(4) = 2 $$，那么特征向量的第 1 维和第 3 维会进行加减后作为新的特征向量的第 1 维。

可以证明，按上面的方式生成的新特征 $$ \vec{y}$$ 在概率意义下保留了原始特征空间的内积，以及距离：

$$
\vec{x}_1^T \cdot \vec{x}_2 \approx \vec{y}_1^T \cdot \vec{y}_2
$$

$$
||\vec{x}_1 - \vec{x}_2|| \approx ||\vec{y}_1 - \vec{y}_2||
$$

其中 $$ \vec{x}_1, \vec{x}_2 $$ 为两个原始特征向量，而 $$ \vec{y}_1, \vec{y}_2 $$为对应的哈希后的特征向量。


sklearn 中 Feature hashing 主要通过类 [sklearn.feature_extraction.FeatureHasher](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html) 实现。

{%ace edit=true, lang='java'%}
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import FeatureHasher

measurements = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]

vec = DictVectorizer()
measurements_vec = vec.fit_transform(measurements)
print(measurements_vec.toarray())
// [[2. 1. 4. 0.]
//  [0. 2. 0. 5.]]

hash = FeatureHasher(n_features=10)
measurements_hash = hash.transform(measurements)
print(measurements_hash.toarray())
// [[ 0.  0. -4. -1.  0.  0.  0.  0.  0.  2.]
//  [ 0.  0.  0. -2. -5.  0.  0.  0.  0.  0.]]
{%endace%}
