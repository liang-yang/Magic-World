<!-- toc -->

# Polynomial Features

---

Polynomial Features(多项式化特征) 是指对当前的特征值取高次，或者将不同维度间的特征值进行组合，形成新的特征值。

$$ (x_1, x_2)^2 \to (1, x_1, x_2, x_1x_2, x_1^2, x_2^2) $$

$$ (x_1, x_2)^3 \to (1, x_1, x_2, x_1x_2, x_1^2x_2, x_1x_2^2, x_1^2, x_2^2, x_1^3, x_2^3) $$

$$ (x_1, x_2, x_3)^2 \to (1, x_1, x_2, x_3, x_1x_2, x_1x_3, x_2x_3, x_1^2, x_2^2, x_3^2) $$

通过此种方式，会增加特征向量的维数，便于寻找更拟合的数据关系。例如，对于 $$y=2x^3 + x^2 - 5x - 1$$ 的回归关系，属于非线性关系，分析起来比较麻烦。但如果我们将特征向量 $$(x)$$ 通过三次多项式转化为 $$(1,x,x^2,x^3)$$，就可以通过线性拟合得到回归关系，计算起来更为简单。

在 sklearn 中通过类 [sklearn.preprocessing.PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) 对特征值多项式化。

{%ace edit=false, lang='java'%}
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.arange(6).reshape(3, 2)
poly = PolynomialFeatures(3)
print(poly.fit_transform(X))
// [[  1.   0.   1.   0.   0.   1.   0.   0.   0.   1.]
//  [  1.   2.   3.   4.   6.   9.   8.  12.  18.  27.]
//  [  1.   4.   5.  16.  20.  25.  64.  80. 100. 125.]]
{%endace%}

