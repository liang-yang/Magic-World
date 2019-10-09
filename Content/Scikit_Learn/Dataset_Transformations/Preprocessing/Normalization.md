<!-- toc -->

# Normalization

---

**归一化** 是将单个样本缩放到单位范式的过程，在 基于 [向量空间模型](https://wiki.mbalib.com/wiki/%E5%90%91%E9%87%8F%E7%A9%BA%E9%97%B4%E6%A8%A1%E5%9E%8B) 的 文本分类和聚类 等计算样本间的相似性时非常有用。

> 在数学领域，范式（Norm）是一个函数，可赋予某向量以长度或大小。
$$n$$ 维向量 $$\vec{x} = (x_1,x_2,...,x_n)$$ 的 $$P$$ 阶范式一般性定义为：$$||\vec{x}||_{P} = (\sum_{i=1}^n ||x_i||^P)^{\frac{1}{P}}$$。      
$$\vec{x}$$ 的 L1 范式表示所有元素绝对值之和，即：$$||\vec{x}||_{1} = (\sum_{i=1}^n ||x_i||)$$；
$$\vec{x}$$ 的 L2 范式表示所有元素平方之和再开二次方，即：$$||\vec{x}||_{2} = \sqrt{\sum_{i=1}^n ||x_i||^2}$$；

在 sklearn 中，可通过函数 [sklearn.preprocessing.normalize](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html) 和类 [sklearn.preprocessing.Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html) 进行 L1、L2 范式的归一化。

{%ace edit=false, lang='java'%}
from sklearn.preprocessing import Normalizer

X = [[4, 1, 2, 2],
     [1, 3, 9, 3],
     [5, 7, 5, 1]]
transformer = Normalizer()
Xn = transformer.fit_transform(X)
print(Xn)
// [[0.8 0.2 0.4 0.4]
// [0.1 0.3 0.9 0.3]
// [0.5 0.7 0.5 0.1]]
{%endace%}

另外，稀疏矩阵也可进行 归一化 处理，不会影响其稀疏性。


