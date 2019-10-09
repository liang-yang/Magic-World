<!-- toc -->

# Validation Curve, Learning Curve

---

see [Validation curves: plotting scores to evaluate models](https://scikit-learn.org/stable/modules/learning_curve.html)

泛化能力 是评价模型优劣的一个非常重要的指标，指标表现有如下 四 种场景：

1. 训练得分高，验证得分高，则估计器拟合的很好；
2. 训练得分高，验证得分低，则估计器过拟合；
3. 训练得分低，验证得分低，则估计器欠拟合；
4. 训练得分低，验证得分高，通常不太可能，即使出现也无意义；

通过以下两个方法可以优化泛化能力：

1. 调整算法模型的超参数；
2. 增加训练集的样本数量；

以上两个方法，分别对应 Validation Curve 和 Learning Curve。

## 1. Validation Curve

使用 GirdSearchCV 分析调整超参数对于提升泛化能力的效果有两点不足：

1. GirdSearchCV 是遍历 超参数组合，对于 泛化能力 的变化无法具体到单个的超参数；
2. GirdSearchCV 未返回训练集 的 score；

sklearn 中，可通过 [sklearn.model_selection.validation_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html) 来分析单个超参数对训练分数和验证分数的影响，所以其参数 param_name 为单个超参数。

> demo see [Plotting Validation Curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py)

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target

param_range = np.logspace(-7, 2, 8)
train_scores, valid_scores = validation_curve(SVC(), X, y, param_name="gamma", param_range=param_range, cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

plt.title("Validation Curve with SVM")
plt.xlabel(r"gamma")
plt.ylabel("Score")
plt.ylim(0.6, 1.1)

lw = 2

plt.semilogx(param_range, train_scores_mean, label="Training Score", color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, valid_scores_mean, label="Cross-Validation Score", color="navy", lw=lw)
plt.fill_between(param_range, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()

print(train_scores_mean)
print(train_scores_std)
print(valid_scores_mean)
print(valid_scores_std)
print(train_scores)
print(valid_scores)
// train_scores_mean
// [0.92       0.92166667 0.92166667 0.92166667 0.95       0.985
//  0.99333333 1.        ]
// train_scores_std
// [0.01130388 0.01130388 0.01130388 0.01130388 0.01394433 0.0062361
//  0.00333333 0.        ]
// valid_scores_mean
// [0.91333333 0.91333333 0.91333333 0.91333333 0.94       0.98
//  0.96666667 0.6       ]
// valid_scores_std
// [0.05416026 0.05416026 0.05416026 0.05416026 0.04422166 0.01632993
//  0.02108185 0.05577734]
// train_scores
// [[0.925      0.91666667 0.93333333 0.925      0.9       ]
//  [0.925      0.925      0.93333333 0.925      0.9       ]
//  [0.925      0.925      0.93333333 0.925      0.9       ]
//  [0.925      0.925      0.93333333 0.925      0.9       ]
//  [0.95       0.95       0.95833333 0.96666667 0.925     ]
//  [0.98333333 0.98333333 0.99166667 0.99166667 0.975     ]
//  [0.99166667 0.99166667 0.99166667 1.         0.99166667]
//  [1.         1.         1.         1.         1.        ]]
// valid_scores
// [[0.86666667 0.96666667 0.83333333 0.96666667 0.93333333]
//  [0.86666667 0.96666667 0.83333333 0.96666667 0.93333333]
//  [0.86666667 0.96666667 0.83333333 0.96666667 0.93333333]
//  [0.86666667 0.96666667 0.83333333 0.96666667 0.93333333]
//  [0.93333333 0.96666667 0.86666667 0.93333333 1.        ]
//  [0.96666667 1.         0.96666667 0.96666667 1.        ]
//  [1.         0.96666667 0.93333333 0.96666667 0.96666667]
//  [0.66666667 0.53333333 0.56666667 0.66666667 0.56666667]]
{%endace%}

![](http://ww1.sinaimg.cn/large/006tNc79gy1g5sc7r15z5j30hs0dc3yq.jpg)

上例中，橙色是训练分数，蓝色是验证分数。我们可以看出：

1. 超参数 gamma 在【$$10^{-6}, 10^{0}$$】范围内，随着 gamma 的增长，训练分数 和 验证分数 均增长，模型拟合的越来越好；
2. 超参数 gamma 在【$$10^{0}, 10^{2}$$】范围内，随着 gamma 的增长，训练分数增长，但 验证分数 快速下降，出现过拟合；

因此，超参数 gamma 的取值在 $$10^{0}$$ 附近泛化能力最好。

## 2. Learning Curve

提升 泛化能力，还可以通过增加训练集样本数量的方式，这就是 **Learning Curve**。但是，我们怎样观察样本数量的增加对指标的影响呢？

sklearn 中，可通过 [sklearn.model_selection.learning_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html) 来观察。

> demo see [Plotting Learning Curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)

下面是两张 Learning Curve，展示了随着样本数量增加，训练得分 和 验证得分 的变化趋势。

![](http://ww1.sinaimg.cn/large/006tNc79gy1g5sfwgs44qj30hs0dcdgb.jpg)

![](http://ww4.sinaimg.cn/large/006tNc79gy1g5sfxbva2ej30hs0dc3yx.jpg)

