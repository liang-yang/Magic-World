<!-- toc -->

# Precision, Recall, F-measures

---

Accuracy 是所有分类类别的预测结果的准确率，而下面讨论的是针对某单独分类类别的指标。

在讨论下面的指标前，我们还需要有一个认识：

**分类模型是基于 判别阈值(Threshold) 进行分类的。**

随着 判别阈值 的变化，样本分类也可能发生变化。同一分类算法，分类 $$c_i$$ 的判别阈值 越高，$$TP_{c_i},FP_{c_i}$$ 越少，$$FN_{c_i},TN_{c_i}$$ 越多。

## 1. Binary Classification

### 1.1. Precision

Precision【精确率/查准率】是指某分类类别的预测结果的准确率，例如分类 $$c_i$$ 的 Precision 为：

$$
Precision_{c_i} = \frac{TP_{c_i}}{TP_{c_i}+FP_{c_i}}
$$

sklearn 中，可通过函数 [sklearn.metrics.precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html) 计算 Precision。

{%ace edit=true, lang='java'%}
from sklearn.metrics import precision_score

y_true = [0, 1, 0, 0, 1, 1, 1]
y_pred = [0, 0, 1, 0, 0, 1, 1]
print(precision_score(y_true, y_pred))
// 0.6666666666666666
{%endace%}

一般来说，Precision 越高，此类别的分类效果越好。但是，如果我们把 $$c_i$$ 的 分类判别阈值 定的很高，只在非常有把握的情况下才将样本判别为 $$c_i$$，这样可以明显提高 Precision。不过，这样我们会遗漏大量的真实分类为 $$c_i$$ 的样本。  
因此，引入了 Recall。

### 1.2. Recall

Recall【召回率/查全率】，顾名思义，是指真实结果为某分类类别的所有样本被准确找到的比例。例如 $$c_i$$ 的 Recall 为：

$$
Recall_{c_i} = \frac{TP_{c_i}}{TP_{c_i}+FN_{c_i}}
$$

sklearn 中，可通过函数 [sklearn.metrics.recall_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html) 计算 Recall。

{%ace edit=true, lang='java'%}
from sklearn.metrics import recall_score

y_true = [0, 1, 0, 0, 1, 1, 1]
y_pred = [0, 0, 1, 0, 0, 1, 1]
print(recall_score(y_true, y_pred))
// 0.5
{%endace%}

一般来说，Recall 越高，此类别的分类效果越好。但是，如果我们把 $$c_i$$ 的 分类判别阈值 定的很低，甚至所有样本均判定为 $$c_i$$，这样 Recall 可以达到很高。但实际上，这样并没有意义。

### 1.3. F-measure

理想情况下，我们希望 Precision 和 Recall 都很高。但实际场景中，两者一般是此消彼长的关系：随着判别阈值的提高，Precision 逐渐提高，Recall 逐渐降低。

因此，为了统一的评估分类模型，我们定义了 F-measure：

$$
F_{\beta} = (1+\beta^2) \cdot \frac{Precision \cdot Recall}{\beta^2 \cdot Precision + Recall}
$$

我们取 $$\beta=1$$，即为：

$$
F_{1} = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

sklearn 中，可通过函数 [sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) 直接计算 $$F_1$$，也可通过函数 [sklearn.metrics.fbeta_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html) 计算 $$F_{\beta}$$。

{%ace edit=true, lang='java'%}
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score

y_true = [0, 1, 0, 0, 1, 1, 1]
y_pred = [0, 0, 1, 0, 0, 1, 1]
print(f1_score(y_true, y_pred))
// 0.5714285714285715
print(fbeta_score(y_true, y_pred, beta=1))
// 0.5714285714285715
{%endace%}

## 2. Multiclass Classification

当在多分类场景下时，就需要对多个分类的 Precision、Recall、F-measure 进行合并。合并的方式通过参数 average 来控制：

- **macro**：计算每个分类的指标值，然后对所有指标值进行算术平均。
- **weighted**：计算每个分类的指标值，然后基于真实分类的数量占比作为权重值来计算加权算术平均。
- **micro**：计算各个分类的TP、FP、TN、FN，然后求和计算总的TP、FP、TN、FN，最后根据公式进行计算。

> 参考 [微平均micro,宏平均macro计算方法](https://www.jianshu.com/p/9e0caf109e88)

可以看出，**micro** 类似于Accuracy Score 的计算方式，**macro** 类似于 Balanced Accuracy Score 的计算方式。

> 注：**micro**计算中，总FP=总FN，因此 Precision 始终等于 Recall。

## 3. Classification Report

我们可以通过函数 [sklearn.metrics.precision_recall_fscore_support](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html) 一次性计算 precision、recall、fscore 和 support(support 是指各个分类真实值为正的样本数量)。

更简单的，可以通过函数 [sklearn.metrics.classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) 生成一份更为直观的报告。

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
print(precision_recall_fscore_support(y_true, y_pred))
// (array([0.66666667, 0.        , 0.        ]), array([1., 0., 0.]), array([0.8, 0. , 0. ]), array([2, 2, 2]))
print(precision_recall_fscore_support(y_true, y_pred, average='macro'))
// (0.2222222222222222, 0.3333333333333333, 0.26666666666666666, None)
print(precision_recall_fscore_support(y_true, y_pred, average='micro'))
// (0.3333333333333333, 0.3333333333333333, 0.3333333333333333, None)
print(precision_recall_fscore_support(y_true, y_pred, average='weighted'))
// (0.2222222222222222, 0.3333333333333333, 0.26666666666666666, None)

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
// precision recall f1-score support
//
// cat 0.67 1.00 0.80 2
// dog 0.00 0.00 0.00 2
// pig 0.00 0.00 0.00 2
//
// accuracy 0.33 6
// macro avg 0.22 0.33 0.27 6
// weighted avg 0.22 0.33 0.27 6
{%endace%}

