<!-- toc -->

# Accuracy Score

---

## 1. Accuracy Score

Accuracy Score（准确率）表示 预测结果与真实结果相同的样本数量 在 总样本 中的占比：

$$
AccuracyScore = \frac{\sum_{i = 1}^{n} TP_{c_i}}{\sum_{i = 1}^{n}(TP_{c_i}+FP_{c_i})}
$$

sklearn 中，可通过函数 [sklearn.metrics.accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) 计算 Accuracy Score。

{%ace edit=true, lang='java'%}
from sklearn.metrics import accuracy_score

y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
print('Accuracy Fraction: ', accuracy_score(y_true, y_pred))
// Accuracy Fraction:  0.5
print('Accuracy Count: ', accuracy_score(y_true, y_pred, normalize=False))
// Accuracy Count:  2
{%endace%}

一般来说，Accuracy Score 能直观的反映预测的优劣。但在各分类分布不均衡的情况下，Accuracy Score 就会失真了。
   
举个医疗领域的例子。假设某病症的发病率为 0.01%，如果分类模型对所有输入样本均返回“未发病”，那么 Accuracy Score 可达到 99.99%。但是，这么“高”的 Accuracy Score 却没有任何意义。   

## 2. Balanced Accuracy Score

Accuracy Score 在分布不均衡的数据集中失真的原因，是由于对所有分类统一分析准确率，这样会掩藏部分分类的“不准确”，尤其是 **样本数量少但非常重要** 的分类。

为了规避此种场景，引入了 Balanced Accuracy Score，原理是针对每种分类单独计算准确率，然后基于算术平均得到 Balanced Accuracy Score：

$$
BalancedAccuracyScore = (\sum_{i=1}^n \frac{TP_{c_i}}{TP_{c_i}+FP_{c_i}})/n
$$

另外，为规避随机性概率的影响，可进行如下优化：

$$
BalancedAccuracyScore_{adjusted} = \frac{BalancedAccuracyScore - \frac{1}{n}}{1 - \frac{1}{n}}
$$

sklearn 中，可通过函数 [sklearn.metrics.balanced\_accuracy\_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html) 计算 Balanced Accuracy Score。

{%ace edit=true, lang='java'%}
from sklearn.metrics import balanced_accuracy_score

y_true = [0, 1, 0, 0, 1, 0, 2]
y_pred = [0, 1, 0, 0, 0, 1, 2]
print(balanced_accuracy_score(y_true, y_pred))
// 0.75
print(balanced_accuracy_score(y_true, y_pred,adjusted=True))
// 0.625
{%endace%}
