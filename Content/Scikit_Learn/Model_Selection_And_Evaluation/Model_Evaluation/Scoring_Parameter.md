<!-- toc -->

# Scoring Parameter

---

see [Scoring Parameter](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules)

Scoring Parameter，是指供 交叉验证( Cross Validation) 和 网格搜索(Grid Search) 使用的评估参数。

## 1. Predefined Values

包 [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) 中预定义了大量的评估指标，如分类模型的 balanced\_accuracy，聚类模型的 completeness\_score，回归模型的 explained\_variance 等，可直接通过名称引用。

## 2. Defining Scoring Strategy With make_scorer 

基于函数 [sklearn.metrics.make_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html) 自定义评估指标。

> 函数以 _score 结尾，表示返回值越大越好；
> 函数以 _error 或 _loss 结尾，表示返回值越小越好；

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()

clf = svm.SVC(kernel='linear', C=1, random_state=0)

f2_scorer = make_scorer(fbeta_score, beta=2, average="macro")


def my_custom_loss_func(y_true, y_pred):
    diff = np.abs(y_true - y_pred).max()
    return np.log1p(diff)


custom_scorer = make_scorer(my_custom_loss_func, greater_is_better=False)

scores = cross_validate(clf, iris.data, iris.target, scoring={"f2": f2_scorer, "custom": custom_scorer}, cv=5)
print(scores)
// {
//     'fit_time': array([0.00106907, 0.0008111, 0.0005157, 0.00089693, 0.00060821]),
//     'score_time': array([0.003654, 0.00135612, 0.00161004, 0.00121403, 0.00143576]),
//     'test_f2': array([0.96625317, 1., 0.96625317, 0.96625317, 1.]),
//     'test_custom': array([-0.69314718, -0., -0.69314718, -0.69314718, -0.])
// }
{%endace%}

## 3. Defining Scoring Strategy Without make_scorer 

如果不基于函数 [sklearn.metrics.make_scorer](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html) 自定义评估指标，则需要满足以下两个条件：

1. 函数的参数 (estimator, X, y)。其中 estimator 是要被评估的模型，X 是验证数据， y 是 X (在有监督情况下) 或 None (在无监督情况下) 已经被标注的真实数据目标；
2. 函数的返回是一个浮点数，表示 estimator 基于 X 训练后得到的预测值与真实值 y 比较的评估结果；

## 4. Multiple Metric Evaluation

交叉验证( Cross Validation) 和 网格搜索(Grid Search) 中，还支持同时指定多个评估指标。

格式有 List 和 Dict 两种：

{%ace edit=true, lang='java'%}
scoring = ['accuracy', 'precision']
{%endace%}

{%ace edit=true, lang='java'%}
scoring = {'acc':'accuracy', 'prec':'precision'}
{%endace%}
