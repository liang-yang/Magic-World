<!-- toc -->

# Dummy Estimators

---

see [Dummy Estimators](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)

虚拟估计模型，分为 虚拟分类模型 和 虚拟回归模型，目的是基于历史经验的简单规则得到一个评估基线，作为与其他真实 分类模型 和 回归模型 比较的依据。

> 需要注意，虚拟估计模型在评估时仅考虑了训练集的结果数据，没有考虑训练集的特征数据。

## 1. DummyClassifier

sklearn 中，类 [sklearn.dummy.DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) 实现了几种简单的分类策略：

- **stratified**：基于训练集的结果分布随机生成结果；
- **most_frequent**：预测值总是训练集中最频繁的分类；
- **prior**：预测值总是最大先验概率的分类 (类似于 most_frequent) 并且 predict_proba 返回最大先验概率；
- **uniform**：随机预测；
- **constant**：预测值总是用户提供的常量；

{%ace edit=true, lang='java'%}
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

iris = load_iris()
X, y = iris.data, iris.target
y[y != 1] = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf_dummy = DummyClassifier(strategy='most_frequent', random_state=0).fit(X_train, y_train)
clf_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
clf_rbf = SVC(gamma='scale', kernel='rbf', C=1).fit(X_train, y_train)

print("dummy clf score: ", clf_dummy.score(X_test, y_test))
print("linear clf score: ", clf_linear.score(X_test, y_test))
print("rbf clf score: ", clf_rbf.score(X_test, y_test))
// dummy clf score:  0.5789473684210527
// linear clf score:  0.631578947368421
// rbf clf score:  0.9473684210526315
{%endace%}

> 上例中，linear分类模型 并不比 虚拟分类模型 好多少，不是一个好的分类模型。

## 2. DummyRegressor

sklearn 中，类 [sklearn.dummy.DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html) 实现了几种简单的回归策略：

- **mean**：平均值；
- **median**：中位数；
- **quantile**：分位数；
- **constant**：用户提供的常量；
