<!-- toc -->

# Other Metrics

---

## 1. Confusion matrix

Confusion matrix(混淆矩阵)，是以矩阵形式将数据集中的记录按照真实的类别与分类模型预测的类别判断两个标准进行汇总。其中矩阵的行表示真实值，矩阵的列表示预测值。

二分类混淆矩阵 |预测值：类别1|预测值：类别2
:-:|:-:|:-:
**真实值：类别1** | $$a$$ | $$b$$
**真实值：类别2** | $$c$$ | $$d$$

多分类混淆矩阵 |预测值：类别1|预测值：类别2|预测值：类别3
:-:|:-:|:-:|:-:
**真实值：类别1** | $$a$$ | $$b$$ | $$c$$ 
**真实值：类别2** | $$d$$ | $$e$$ | $$f$$ 
**真实值：类别3** | $$g$$ | $$h$$ | $$i$$ 

sklearn 中，可通过函数 [sklearn.metrics.confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) 计算 混淆矩阵：

{%ace edit=true, lang='java'%}
from sklearn.metrics import confusion_matrix

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
print(confusion_matrix(y_true, y_pred))
// [[2 0 0]
//  [0 0 1]
//  [1 0 2]]
{%endace%}

## 2. Cohen Kappa

> see [诊断试验之Kappa值该怎么算？](https://zhuanlan.zhihu.com/p/25973834)

Cohen 于1960年提出 Kappa 分析，是在**消除 随机影响** 后评估 分类结果，其公式为：

$$
Kappa = \frac{p_0 - p_e}{1 - p_e}
$$

其中，$$p_0$$ 是指表面准确率，$$p_e$$ 是指 随机准确率。
 
下面举一个案例。两位病理科医生对75位患者诊断结果的混淆矩阵如下：

混淆矩阵 |医生A：类别1|医生A：类别2
:-:|:-:|:-:
**医生B：类别1** | $$41$$ | $$3$$
**医生B：类别2** | $$4$$ | $$27$$

表面准确率 $$p_0 = \frac{41+27}{75} = 90.7\% $$。

试验中，医生A认定这75位患者中有 60% 为类别1，40% 为类别2。我们假设这是医生A的随机判定概率。那么，医生A会认为44位被医生B判断为类别1中有60%（26.4）为类别1，40%（17.6）为类别2。同理，医生A也会认为31位被医生B判断是类别2中有60%（18.6）为类别1，40%（12.4）为类别2。那么，医生A随机判定后形成的混淆矩阵如下：

混淆矩阵|医生A：类别1|医生A：类别2
:-:|:-:|:-:
**医生B：类别1** | $$26.4$$ | $$17.6$$
**医生B：类别2** | $$18.6$$ | $$12.4$$

随机准确率 $$p_e = \frac{26.4+12.4}{75} = 51.7\%$$。

最终，有：

$$
Kappa = \frac{90.7\% - 51.7\%}{1 - 51.7\%} = \frac{39\%}{48.3\%} = 0.81
$$

一般来说，Kappa 大于 0.8 就说明分类结果是比较可信的，否则就有较大概率是由于随机因素影响的。

sklearn 中，可通过函数 [sklearn.metrics.cohen_kappa_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html) 计算 kappa值：

{%ace edit=true, lang='java'%}
from sklearn.metrics import cohen_kappa_score

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
print(cohen_kappa_score(y_true, y_pred))
// 0.4285714285714286
{%endace%}

## 3. Jaccard Similarity Coefficient

Jaccard Similarity Coefficient（Jaccard相似度系数）用于比较有限样本集之间的相似性与差异性。Jaccard系数值越大，样本相似度越高。

给定两个集合 A,B，Jaccard相似度系数 定义为 A 与 B 交集的大小与 A 与 B 并集的大小的比值，定义如下：

$$
J(A,B) = \frac{|A \cap B|}{|A \cup B|}
$$

> 当集合 A,B 都为空时，$$J(A,B)$$ 定义为 1。

在分类问题上，Jaccard相似度系数 与 Accuracy 相同。

sklearn 中，可通过函数 [sklearn.metrics.jaccard_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html) 计算 Jaccard Similarity Coefficient 值：

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.metrics import jaccard_score

y_true = np.array([[0, 1, 1], [1, 1, 0]])
y_pred = np.array([[1, 1, 1], [1, 0, 0]])
print(jaccard_score(y_true[0], y_pred[0]))
// 0.6666666666666666
{%endace%}

## 4. Matthews Correlation Coefficient

MCC（Matthews correlation coefficient），马修斯相关系数，常用以测量二分类的分类性能的指标。该指标考虑了真阳性、真阴性和假阳性和假阴性，**通常认为该指标是一个比较均衡的指标，即使是在两类别的样本含量差别很大时，也可以应用它**。

MCC本质上是一个描述实际分类与预测分类之间的相关系数，它的取值范围为[-1,1]：
- 取值为1时表示对受试对象的完美预测；
- 取值为0时表示预测的结果类似于随机预测的结果；
- 取值为-1时表示预测分类和实际分类完全相反；

sklearn 中，可通过函数 [sklearn.metrics.matthews_corrcoef](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html) 计算 Matthews Correlation Coefficient 值：

{%ace edit=true, lang='java'%}
from sklearn.metrics import matthews_corrcoef

y_true = [+1, +1, +1, -1]
y_pred = [+1, -1, +1, +1]
print(matthews_corrcoef(y_true, y_pred))
// -0.3333333333333333
{%endace%}

## 5. Brier Score

see [概率校准与Brier分数](https://www.cnblogs.com/sddai/p/9581142.html)

Brier Score 可以被认为是对一组概率预测的量度：一般Brier分数越低，表示预测越准。

在二分类中，公式如下：

$$
Brier Score = \frac{1}{N} \sum^N_{i=1}(Pp_{i} - Pt_i)^2
$$

其中，$$Pp_{i}$$ 是预测的概率，$$Pt_i$$真实发生的概率（$$Pt_i$$要么为0，要么为1）。

> 举例说明。假设一个人预测在某一天会下雨的概率P，则Brier分数计算如下：  
如果预测为100％（$$P_p$$ = 1），并且下雨（$$P_t$$ = 1），则Brier Score为0，达到最佳分数。  
如果预测为100％（$$P_p$$ = 1），但是不下雨（$$P_t$$ = 0），则Brier Score为1，可达到最差分数。  
如果预测为70％（$$P_p$$ = 0.70），并且下雨（$$P_t$$ = 1），则Brier评分为 0.09。  
如果预测为30％（$$P_p$$ = 0.30），并且下雨（$$P_t$$ = 1），则Brier评分为 0.49。  
如果预测为50％（$$P_p$$ = 0.50），则Brier分数为 0.25，无论是否下雨。

sklearn 中，可通过函数 [sklearn.metrics.brier_score_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html) 计算 Brier Score：

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.metrics import brier_score_loss

y_true = np.array([0, 1, 1, 0])
y_true_categorical = np.array(["spam", "ham", "ham", "spam"])
y_prob = np.array([0.1, 0.9, 0.8, 0.4])
y_pred = np.array([0, 1, 1, 0])

print(brier_score_loss(y_true, y_prob))
// 0.055
print(brier_score_loss(y_true, 1 - y_prob, pos_label=0))
// 0.055
print(brier_score_loss(y_true_categorical, y_prob, pos_label="ham"))
// 0.055
print(brier_score_loss(y_true, y_prob > 0.5))
// 0
{%endace%}
