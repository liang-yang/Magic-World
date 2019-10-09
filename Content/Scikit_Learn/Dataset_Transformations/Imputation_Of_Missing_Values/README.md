<!-- toc -->

# Imputation Of Missing Values

---

see [Imputation Of Missing Values](https://scikit-learn.org/stable/modules/impute.html)

由于各种原因，许多实际数据集包含缺失的值，这些值通常被编码为空格、NaNs或其他占位符。   
然而，这些数据集与 scikit-learn 估计器不兼容，后者假定数组中的所有值都是数值，并且所有值都具有意义。   
使用不完整数据集的基本策略是丢弃包含丢失值的整个行或列。然而，这是以丢失数据为代价的，这些数据可能是有价值的(即使是不完整的)。 处理缺失数值的一个更好的策略就是从已有的数据推断出缺失的数值。

在 sklearn 中通过类 [sklearn.impute.SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) 提供了多种计算缺失值的方法：

- **mean**：平均值；
- **median**：中值；
- **most_frequent**：众值；
- **constant**：某常量；

下例中通过 fit 得到的均值来 transform 后面的数据：

{%ace edit=false, lang='java'%}
import numpy as np
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit([[1, 2], [np.nan, 3], [7, 6]])

X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))
// [[4.         2.        ]
//  [6.         3.66666667]
//  [7.         6.        ]]
{%endace%}

同样的，SimpleImputer 也支持稀疏矩阵。

另外，我们可以通过 [sklearn.impute.MissingIndicator](https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html) 将缺失值转化为布尔矩阵。

{%ace edit=false, lang='java'%}
import numpy as np
from sklearn.impute import MissingIndicator

X = np.array([[-1, -1, 1, 3],
              [4, -1, 0, -1],
              [8, -1, 1, 0]])
indicator = MissingIndicator(missing_values=-1, features="all")
print(indicator.fit_transform(X))
// [[ True  True False False]
//  [False  True False  True]
//  [False  True False False]]
{%endace%}
