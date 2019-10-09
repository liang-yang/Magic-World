<!-- toc -->

# Cross-Validation Computing

---

Cross-Validation Computing，数据集 split 后，在 训练集 上训练模型，在 验证集 上评估模型，输出各次迭代的评估指标值。

## 1. Train Test Split

sklearn 中，可通过函数 [sklearn.model\_selection.train\_test\_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) 随机的将样本集分为 训练集 和 测试集。我们可以基于 训练集 训练模型，然后通过 测试集 验证模型：

{%ace edit=true, lang='java'%}
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

print("dataset size: ", iris.data.shape)
print("train_set size: ", X_train.shape)
print("test_set size: ", X_test.shape)
// dataset size:  (150, 4)
// train_set size:  (90, 4)
// test_set size:  (60, 4)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print("test score: ", clf.score(X_test, y_test))
// test score:  0.9666666666666667
{%endace%}

## 2. Cross-Validation Computing

### 2.1. cross-validate-score

使用交叉验证最简单的方法是在 估计器(Estimator) 和 数据集 上调用函数 [sklearn.model_selection.cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)。

此函数有两个核心参数：

1. **cv**：指定 Cross Validation 的 split 算法。
    - 输入 integer 或 None。 此时使用 StratifiedKFold 或 KFold Split（分类模型使用StratifiedKFold，其余使用KFold）split；
    - 输入 CV Iterator（交叉验证迭代器）split；
    - 输入 iterable yielding（可迭代生成器）split；

2. **scoring**：指定 Cross Validation 的 评估指标。
    - 估计器(Estimator)一般都会带有 score 函数，作为默认的评估指标；
    - 指标函数(Metric Functions)：在 [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) 包中定义了大量的指标函数，可通过字符串映射指定；

下例中，SVM 模型在 iris 数据集采用 不同的 split 方案和 score 指标：

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

iris = datasets.load_iris()
clf = svm.SVC(kernel='linear', C=1)

ss_cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)


def custom_cv_2folds(X):
    n = X.shape[0]
    i = 1
    while i <= 2:
        idx = np.arange(n * (i - 1) / 2, n * i / 2, dtype=int)
        yield idx, idx
        i += 1


cv_default_split_default_scores = cross_val_score(clf, iris.data, iris.target, cv=5)
cv_default_split_f1_macro_scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_macro')
cv_shuffle_split_default_scores = cross_val_score(clf, iris.data, iris.target, cv=ss_cv)
cv_yield_split_default_scores = cross_val_score(clf, iris.data, iris.target, cv=custom_cv_2folds(iris.data))

print("cv_default_split_default_scores:", cv_default_split_default_scores)
print("cv_default_split_f1_macro_scores:", cv_default_split_f1_macro_scores)
print("cv_shuffle_split_default_scores:", cv_shuffle_split_default_scores)
print("cv_yield_split_default_scores:", cv_yield_split_default_scores)
// cv_default_split_default_scores: [0.96666667 1.         0.96666667 0.96666667 1.        ]
// cv_default_split_f1_macro_scores: [0.96658312 1.         0.96658312 0.96658312 1.        ]
// cv_shuffle_split_default_scores: [0.97777778 0.97777778 1.         0.95555556 1.        ]
// cv_yield_split_default_scores: [1.         0.97333333]
{%endace%}

### 2.2. cross-validate

函数 [sklearn.model_selection.cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) 与 cross\_val\_score 基本一致，但有两点差别：

1. cross\_validate 的 scoring 参数支持指定多个评估指标；
2. cross\_validate 返回的是一个字典，包含：fit-times(训练时间)、score-times(评估时间)、test\_score(多个评估指标就会有多个test\_score)；

{%ace edit=true, lang='java'%}
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn import datasets

iris = datasets.load_iris()

scoring = ['precision_macro', 'recall_macro']
clf = svm.SVC(kernel='linear', C=1, random_state=0)
scores = cross_validate(clf, iris.data, iris.target, scoring=scoring, cv=5)

print(scores)
// {
//     'fit_time': array([0.00366974, 0.00115013, 0.00101399, 0.00068283, 0.00058818]),
//     'score_time': array([0.00338912, 0.00300097, 0.00213575, 0.00185513, 0.00165677]),
//     'test_precision_macro': array([0.96969697, 1., 0.96969697, 0.96969697, 1.]),
//     'test_recall_macro': array([0.96666667, 1., 0.96666667, 0.96666667, 1.])
// }
{%endace%}

### 2.3. cross-validate-predict

函数 [sklearn.model_selection.cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) 和 [sklearn.model_selection.cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) 主要是用来评估 训练集 上训练出的模型在 验证集 上的评估指标，而 [sklearn.model_selection.cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) 主要是输出 训练集 上训练出的模型对 验证集 数据的预测值。因此，[sklearn.model_selection.cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) 不需要 scoring 参数。

{%ace edit=true, lang='java'%}
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

iris = load_iris()
clf = SVC(kernel='linear', C=1, random_state=0)
predictions = cross_val_predict(clf, iris.data, iris.target, cv=10)

print("predictions result:")
print(predictions)
print("accuracy score:", metrics.accuracy_score(iris.target, predictions))
// predictions result:
// [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 2 1
//  1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 1 2 2 2 2
//  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
//  2 2]
// accuracy score: 0.9733333333333334
{%endace%}

