<!-- toc -->

# PRC vs ROC

---

## 1. Precision Recall Curve(PRC)

> see [Precision-Recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)

### 1.1. Precision Recall Curve(PRC)

之前提到，**分类模型是基于 判别阈值(Threshold) 进行分类的。** 随着 判别阈值 的变化，分类模型的表现也会相应的发生变化。因此，我们需要有一个方案来动态的衡量分类模型的表现。而 PRC 就是其中一种方案。

PRC 是指以 Precision 为纵轴，Recall 为横轴，将每个样本的值（如概率）作为判别阈值，多个样本形成多个PR点，进而形成 PRC 曲线。如下图所示：

![](http://ww4.sinaimg.cn/large/006tNc79gy1g4mhqjo690j30hs0dcmx7.jpg)

> 需要注意，PRC 曲线的自变量是 判别阈值，更准确的说是样本的分类指标。

我们分析PRC曲线的四个边界点：

1. **(0,0)**：Recall 为 0，Precision 为 0，说明此分类模型特别不好，一个正确的分类都没有找到；
2. **(0,1)**：Recall 为 0，Precision 为 1，说明此分类模型的 判定阈值 特别高，几乎全部判定为负样本，这样即使准确率高也不好；
3. **(1,0)**：Recall 为 1，Precision 为 0，说明此分类模型的 判定阈值 特别低，几乎将所有样本都归为正样本，这样即使查全率高也不好；
4. **(1,1)**：Recall 为 1，Precision 为 1，说明此分类模型非常好，分类又全又准；

因此：

1. 单条PRC曲线中，越靠近右上角 **(1,1)** 的点的 判别阈值 对应的 评估指标 综合性更好；
2. 多条PRC曲线中，越在上的曲线对应的 分类模型 更好。这是因为上方的曲线对同样的 Precision 有更高的 Recall，对同样的 Recall 有更高的 Precision；
3. 相对平滑的曲线更为稳定，不会因 判别阈值 的轻微变化导致 评估指标 剧烈抖动；

sklearn 中，可通过函数 [sklearn.metrics.precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html) 返回 (Precision, Recall, Thresholds) 的三元组：

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.metrics import precision_recall_curve

y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

print(precision)
// [0.66666667 0.5        1.         1.        ]
print(recall)
// [1.  0.5 0.5 0. ]
print(thresholds)
// [0.35 0.4  0.8 ]
{%endace%}

### 1.2. Average Precision(AP)

观察 PRC 曲线较为抽象，既然认为上方的曲线更好，那么很容易想到以 PRC 曲线下的面积作为指标进行判定，这就是 AP(Average Precision)，这里的 Average 相当于是对不同阈值下的 Precision 取均值。

sklearn 中，可通过函数 [sklearn.metrics.average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) 计算AP：

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.metrics import average_precision_score

y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print(average_precision_score(y_true, y_scores))
// 0.8333333333333333
{%endace%}

## 2. Receiver Operating Characteristic(ROC)

> see [ROC & AUC](http://alexkong.net/2013/06/introduction-to-auc-and-roc/)

### 2.1. TPR, FPR

- **TPR** : True Positive Rate，所有正样本中被预测为正的比例，也就是 Recall

$$
TPR_{c_i} = \frac{TP_{c_i}}{TP_{c_i} + FN_{c_i}} = Recall_{c_i}
$$

- **FPR** : False Positive Rate，所有负样本中被预测为正的比例

$$
FPR_{c_i} = \frac{FP_{c_i}}{TN_{c_i} + FP_{c_i}}
$$

我们希望 TPR 越大越好，FPR 越小越好。

### 2.2. Receiver Operating Characteristic(ROC)

ROC 曲线以 TPR 为纵轴，FPR 为横轴，将每个样本的值（如概率）作为判别阈值，多个样本形成 ROC 曲线，如下图所示：

![](http://ww3.sinaimg.cn/large/006tNc79gy1g4mj1sdlyzj30hs0dcq36.jpg)

我们分析 ROC 曲线的四个边界点：

1. **(0,0)**：FPR 为 0，TPR 为 0，说明此分类模型的 判定阈值 特别高，全部判定为负样本；
2. **(0,1)**：FPR 为 0，TPR 为 1，说明此分类模型非常好，分类又全又准；
3. **(1,0)**：FPR 为 1，TPR 为 0，说明此分类模型特别不好，一个正确的分类都没有找到；
4. **(1,1)**：FPR 为 1，TPR 为 1，说明此分类模型的 判定阈值 特别低，全部判定为正样本；

因此：

1. 单条 ROC 曲线中，越靠近左上角(0,1)的点的 判别阈值 对应的 评估指标 综合性更好；
2. 多条 ROC 曲线中，越在上的曲线对应的 分类模型 更好。这是因为上方的曲线对同样的 TPR 有更小的 FPR，对同样的 FPR 有更高的 TPR；
3. 相对平滑的曲线更为稳定，不会因 判别阈值 的轻微变化导致 评估指标 剧烈抖动；

sklearn 中，可通过函数 [sklearn.metrics.roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) 返回 (FPR, TPR, Thresholds) 的三元组：

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn import metrics

y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)

print(fpr)
// [0.  0.  0.5 0.5 1. ]
print(tpr)
// [0.  0.5 0.5 1.  1. ]
print(thresholds)
// [1.8  0.8  0.4  0.35 0.1 ]
{%endace%}

### 2.3. Area Under ROC Curve(AUC)

和 AP 类似，AUC 是指 ROC 曲线下的面积。

sklearn 中，可通过函数 [sklearn.metrics.roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) 计算 AUC：

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.metrics import roc_auc_score

y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print(roc_auc_score(y_true, y_scores))
// 0.75
{%endace%}

## 3. PRC vs ROC

> see [ROC vs PRC](https://blog.csdn.net/pipisorry/article/details/51788927)

### 3.1. ROC 与 PRC 的评估结果一致

同一数据集，不同分类算法的 ROC 和 PRC 的优劣一致：如果A算法的 ROC 优于B算法，那么A算法的 PRC 也一定优于B算法，反之亦然。

> see 《The Relationship Between Precision-Recall and ROC Curves》

### 3.2. ROC 比 PRC 更平滑

通过下图可以看出，ROC 始终很平滑，PRC 可能剧烈变化（尤其是数据集分类比例不均衡的情况）。
   
> - (a)和(c)为 ROC 曲线，(b)和(d)为 PRC 曲线；
> - (a)和(b)的样本集中正负样本数量比例为1：1；
> - (c)和(d)的样本集中正负样本数量比例为1：10；

![](https://ws1.sinaimg.cn/large/006tNbRwgy1fxgwa3ni9cj30hc0gwq3g.jpg)

这是由两个曲线不同的组成指标的公式决定的：
   
1. TPR/Recall 的分母为“真实正样本的数量”，一旦样本集确定，就不再因 判别阈值 的变化而变化。分子为“预测准确的正样本的数量”，随着 判别阈值 的变化单调变化，比较平滑；
2. FDR 的分母为“真实负样本的数量”，一旦样本集确定，就不再因 判别阈值 的变化而变化。分子为“预测错误的负样本的数量”，随着 判别阈值 的变化单调变化，比较平滑；
3. Precision 的分母为“预测为正样本的数量”，分子为“预测准确的正样本的数量”，都会随着判别阈值 的变化而变化，因此 Precision 的变化整体不单调，变化比较剧烈。

因此，ROC 比 PRC 更平滑。

### 3.3. ROC 不一定比 PRC 更准确

需要注意，更平滑仅说明变化趋势较缓，不代表更准确。例如左图中的红点，ROC 很靠近（0,1）点，说明分类模型很好；右图中的红点，PRC 离（1,1）点比较远，说明分类模型不太好。而两个曲线中的 红点，其实是同一个点。

![](https://ws1.sinaimg.cn/large/006tNbRwgy1fxhtqt8s2ij30go085jrm.jpg)

反应到数据上：

$$
\begin{matrix}
TPR = Recall = 0.8 \\
Precision = 0.05 \\
FPR = 0.1
\end{matrix}
$$

我们假设 真实正样本数量 为 100：

$$
\begin{matrix}
TP + FN = 100 \\
TPR = 0.8 \to TP = 80, FN = 20 \\
Precision = 0.05 \to FP = 1520 \\
FPR = 0.1 \to TN = 13680
\end{matrix}
$$

即数据集中真实有 100 个正样本，15200 个负样本。红点 处将 1520 + 80 = 1600 个样本判定为正样本，其中有 80 个正确。

那么，为什么 ROC 和 PRC 会有差异呢？这是由于 PRC 的 Precision、Recall 都是针对我们在意的分类类别在分析，例如上例就是仅分析 正样本 的分类表现，不在意 负样本 的分类表现。而 ROC 的 TPR、FPR 综合考虑了两种分类类别的分析。 

所以，我们最好两条曲线都进行观察，然后针对不同的场景进行选择。
