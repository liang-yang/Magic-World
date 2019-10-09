<!-- toc -->

# Custom Transformers

---

在 sklearn 中，可通过 [sklearn.preprocessing.FunctionTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html) 自定义转换函数。

例如下例中是对特征值取对数：

{%ace edit=false, lang='java'%}
import numpy as np
from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(np.log1p, validate=True)
X = np.array([[0, 1], [2, 3]])
print(transformer.fit_transform(X))
// [[0.         0.69314718]
//  [1.09861229 1.38629436]]
{%endace%}

> 需要注意，定义时最好能带上逆函数，例如 np.log1p 的逆函数为 np.expm1，否则可能触发告警。





