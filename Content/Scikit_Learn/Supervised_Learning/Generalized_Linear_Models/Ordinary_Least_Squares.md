<!-- toc -->

# Ordinary Least Squares

---

see [Ordinary Least Squares](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)

这里的标题是 Ordinary Least Squares，但实际是指基于 最小二乘法 求解常规 Linear Regression 模型。

Linear Regression，是指如下形式的线性模型：

$$
\hat{y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p = X \cdot \vec{w}
$$

最小二乘法，是指数据集实际值和预测值（估计值）之间的残差平方和（residual sum of squares）最小，即：

$$
\min_w ||X \cdot \vec{w} - y||_2^2
$$

以此求解参数 $$\vec{w}$$。

> $$||x||_2$$ 表示向量 $$x$$ 的 L2-norm。

sklearn 中，可通过 [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) 求解线性模型。

{%ace edit=true, lang='java'%}
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

print(reg.coef_)
// [0.5 0.5]
print(reg.intercept_)
// 1.1102230246251565e-16
{%endace%}

使用 最小二乘法 求解 线性回归模型，有一个前提条件：特征值之间相互独立。如果不独立，就会出现多重共线性的问题，导致模型出现偏差。

最小二乘法是通过矩阵 $$X$$ 的奇异值分解来求解的。   
我们假设 $$X$$ 为一个 $$(n\_samples, n\_features)$$ 的矩阵，并且 $$n\_samples > n\_features$$，那么最小二乘法的算法复杂性为 $$O(n\_samples * n\_fearures * n\_fearures)$$。

