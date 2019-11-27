<!-- toc -->

# Linear Model


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

之前我们提到，普通的线性回归在 多重共线性 的场景下会出现问题。实际上，根据 泰勒变换 可以知道，只要多项式次数够多，总是可以完全 fit 样本数据集。
但是，这种完全fit会将所有系统误差全部考虑进去，会出现过拟合，泛化能力很差。 因此，为规避 过拟合 的问题，其中一种方式是减少 线性回归 中系数 $$w$$ 的维度。

**岭回归** 就是基于这个思路来设计的：在普通线性回归的 Loss Function（残差平方和）中添加对于系数 $$w$$ 的惩罚项。

$$
\min_{w} \{ ||X \cdot \vec{w} - y||_2^2 + \alpha \cdot ||w||_2^2 \}
$$

> $$||x||_2$$ 表示向量 $$x$$ 的 L2-norm。

其中， $$\alpha \geq 0$$ 是控制系数收缩量的复杂性参数：$$ \alpha $$ 的值越大，收缩量越大，模型的鲁棒性也更强。







[机器学习算法实践-标准与局部加权线性回归](https://zhuanlan.zhihu.com/p/30422174)

[机器学习算法实践-岭回归和LASSO](https://zhuanlan.zhihu.com/p/30535220)

[坐标上升算法](http://pytlab.github.io/2017/09/01/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95%E5%AE%9E%E8%B7%B5-SVM%E4%B8%AD%E7%9A%84SMO%E7%AE%97%E6%B3%95/)

## Reference

- https://scikit-learn.org/stable/modules/linear_model.html





