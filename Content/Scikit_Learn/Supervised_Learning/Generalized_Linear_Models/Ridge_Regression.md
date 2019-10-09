<!-- toc -->

# Ridge Regression

---

see [Ridge Regression](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)

之前我们提到，普通的线性回归在 多重共线性 的场景下会出现问题。实际上，根据 泰勒变换 可以知道，只要多项式次数够多，总是可以完全 fit 样本数据集。   
但是，这种完全fit会将所有系统误差全部考虑进去，会出现过拟合，泛化能力很差。 因此，为规避 过拟合 的问题，其中一种方式是减少 线性回归 中系数 $$w$$ 的维度。

**岭回归** 就是基于这个思路来设计的：在普通线性回归的 Loss Function（残差平方和）中添加对于系数 $$w$$ 的惩罚项。

$$
\min_{w} \{ ||X \cdot \vec{w} - y||_2^2 + \alpha \cdot ||w||_2^2 \}
$$

> $$||x||_2$$ 表示向量 $$x$$ 的 L2-norm。

其中， $$\alpha \geq 0$$ 是控制系数收缩量的复杂性参数：$$ \alpha $$ 的值越大，收缩量越大，模型的鲁棒性也更强。

sklearn 中，可通过 [sklearn.linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) 拟合岭回归模型。

{%ace edit=true, lang='java'%}
from sklearn.linear_model import Ridge

reg = Ridge(alpha=.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])

print(reg.coef_)
// [0.34545455 0.34545455]
print(reg.intercept_)
// 0.13636363636363638
{%endace%}

关于超参数 $$\alpha$$ 的选择，sklearn 提供了[sklearn.linear_model.RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html) 直接进行交叉验证。其默认采用了 LOO（Leave One Out）交叉验证方案，但用户可通过指定参数 cv 的方式选择交叉验证方案，如 cv=10 将触发10折交叉验证。

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.linear_model import RidgeCV

reg = RidgeCV(alphas=np.logspace(-6, 6, 13))
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])

print(reg.alpha_)
// 0.01
{%endace%}



