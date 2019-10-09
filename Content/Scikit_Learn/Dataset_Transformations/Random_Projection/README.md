<!-- toc -->

# Random Projection

---

see [Random Projection](https://scikit-learn.org/stable/modules/random_projection.html)

Random Projection（随机投影）是一种高维数据的降维方案，其理论基础是 Johnsen-lindenstrauss(J-L)定理，适用于 **基于样本点间距离** 构建的算法模型。

## 1. J-L定理

see [Johnsen-lindenstrauss定理](https://blog.csdn.net/CHIERYU/article/details/65450249)、[J-L定理](https://www.douban.com/note/162173024/)

J-L定理是指：

**一个 $$d$$ 维空间中的 $$n$$ 个点可以近似等距地嵌入到一个 $$k \approx O(\log n)$$ 维的空间。**

其中，**近似等距** 是指保持任意两个点之间的相对远近关系。

> 这个定理的直观理解，等价于一个基本的概率事实：  
一个随机的 $$M$$ 维单位向量到一个随机的 $$D$$ 维子空间上的投影的长度几乎约等于 $$\frac{D}{M}$$。

J-L定理的数学描述为：

**对于任意 $$R^d$$ 空间中的 $$n$$ 个点构成的集合 $$V$$，始终存在一个映射 $$f:R^d \to R^k$$ 使得对任意的 $$u,v \in V$$，有：**

$$
1 - \varepsilon \leqslant \frac{||f(u) - f(v)||^2}{||u - v||^2} \leqslant 1 + \varepsilon
$$

其中，$$0 < \varepsilon < 1$$ 表示误差比。那么，对于映射后空间维度的 $$k$$，有：

$$
k \geqslant 4 \cdot \log n \cdot (\frac{\varepsilon^2}{2} - \frac{\varepsilon^3}{3}) ^ {-1}
$$

这样，就可以计算出在保证一定距离误差比的情况下可映射的最小空间维度。

> 从以上的公式可以看出：  
1. $$k$$ 与 $$d$$ 在计算上基本没有关系，但一般来说 $$d$$ 远大于 $$k$$；
2. 在距离误差比 $$\varepsilon$$ 不变的情况下，样本数量 $$n$$ 越小，$$k$$ 越小；
3. 定理中提到的距离为 欧氏距离；

在 sklearn 中，可通过函数 [sklearn.random\_projection.johnson\_lindenstrauss\_min\_dim](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.johnson_lindenstrauss_min_dim.html) 来计算随机投影的最小维度。

{%ace edit=false, lang='java'%}
from sklearn.random_projection import johnson_lindenstrauss_min_dim

print(johnson_lindenstrauss_min_dim(n_samples=1e6, eps=0.5))
// 663
print(johnson_lindenstrauss_min_dim(n_samples=1e6, eps=[0.5, 0.1, 0.01]))
// [    663   11841 1112658]
print(johnson_lindenstrauss_min_dim(n_samples=[1e4, 1e5, 1e6], eps=0.1))
// [ 7894  9868 11841]
{%endace%}

## 2. 高斯随机投影

有了 J-L定理 的理论基础，接下来就是确定映射 $$f:R^d \to R^k$$ 了。最常见就是随机投影。    
设随机生成一个 $$k \times d$$ 的矩阵 $$A$$。对于 $$\forall v \in R^d$$，左乘矩阵 $$A$$，然后乘以系数 $$\sqrt{\frac{d}{k}}$$，这是有 $$\sqrt{\frac{d}{k}} \cdot A \cdot v \in R^k$$。

> 乘以 $$\sqrt{\frac{d}{k}}$$ 是为了保证 $$ ||\sqrt{\frac{d}{k}} \cdot A \cdot v|| $$ 的数学期望为 $$||v||$$。

在 sklearn 中，可通过类 [sklearn.random\_projection.GaussianRandomProjection](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html) 实现从正态分布 $$N(0,\frac{1}{k})$$ 中随机抽样构成矩阵。

{%ace edit=false, lang='java'%}
import numpy as np
from sklearn import random_projection

X = np.random.rand(100, 10000)
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.shape)
// (100, 3947)
{%endace%}

## 3. 稀疏随机投影

高斯随机投影矩阵的计算量依然很大，因此我们可以使用稀疏随机矩阵来替代，既保证了相似的映射质量，同时具有更高的内存效率和更快的计算速度。

在 sklearn 中，可通过类 [sklearn.random\_projection.SparseRandomProjection](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.SparseRandomProjection.html) 实现稀疏随机投影。

{%ace edit=false, lang='java'%}
import numpy as np
from sklearn import random_projection

X = np.random.rand(100, 10000)
transformer = random_projection.SparseRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.shape)
// (100, 3947)
{%endace%}
