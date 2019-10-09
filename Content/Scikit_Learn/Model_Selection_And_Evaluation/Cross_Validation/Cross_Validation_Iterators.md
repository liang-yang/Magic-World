<!-- toc -->

# Cross-Validation Iterators

---

Cross-Validation Iterators，是指数次迭代将样本集划分为 训练集 和 验证集。

## 1. CV For i.i.d. Data

i.i.d（Independent and Identically Distributed）是指样本数据来源于固定的独立分布，并且数据之间无记忆性。此种类型的数据可使用如下交叉验证方法。

### 1.1. K-Fold

K-Fold：

- 将样本集划分为 K 份；
- 循环 K 次，每次将其中 1 份作为验证集，另外 K-1 份作为训练集；

sklearn 中，可通过类 [sklearn.model_selection.KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) 实现 K-Fold。

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.model_selection import KFold

kf = KFold(n_splits=3, shuffle=True, random_state=1)
print(kf)
// KFold(n_splits=3, random_state=1, shuffle=True)

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [3, 5], [5, 2]])
y = np.array([1, 2, 3, 4, 5, 1])

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
// TRAIN: [0 3 4 5] TEST: [1 2]
// TRAIN: [1 2 3 5] TEST: [0 4]
// TRAIN: [0 1 2 4] TEST: [3 5]
{%endace%}

### 1.2. Repeated K-Fold

Repeated K-Fold，是指重复进行多次 K-Fold。

sklearn 中，可通过类 [sklearn.model_selection.RepeatedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html) 实现 Repeated K-Fold。

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=1)

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [3, 5], [3, 5]])
y = np.array([1, 2, 3, 4, 5, 1])

for train_index, test_index in rkf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
// 第一轮：
// TRAIN: [0 3 4 5] TEST: [1 2]
// TRAIN: [1 2 3 5] TEST: [0 4]
// TRAIN: [0 1 2 4] TEST: [3 5]
// 第二轮：
// TRAIN: [0 1 3 5] TEST: [2 4]
// TRAIN: [1 2 3 4] TEST: [0 5]
// TRAIN: [0 2 4 5] TEST: [1 3]
{%endace%}

### 1.3. Leave One Out (LOO)

Leave One Out (LOO)，留一交叉验证，每一轮仅将单一样本作为 验证集，其余全部作为 训练集。因此，数据量为 N 的样本集就会交叉验证 N 次。

- LOO 相当于 K = 1 的 K-Fold CV；
- LOO 交叉验证次数更多，计算成本会比较高；
- LOO 的验证评估往往会有比较大的方差，这是由于验证评估主要取决于单一的验证样本；

一般来说，5-Fold 或 10-Fold CV 优于 LOO。

sklearn 中，可通过类 [sklearn.model_selection.LeaveOneOut](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html) 实现 Leave One Out。

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [3, 5], [3, 5]])
y = np.array([1, 2, 3, 4, 5, 1])

for train_index, test_index in loo.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
// TRAIN: [1 2 3 4 5] TEST: [0]
// TRAIN: [0 2 3 4 5] TEST: [1]
// TRAIN: [0 1 3 4 5] TEST: [2]
// TRAIN: [0 1 2 4 5] TEST: [3]
// TRAIN: [0 1 2 3 5] TEST: [4]
// TRAIN: [0 1 2 3 4] TEST: [5]
{%endace%}

### 1.4. Leave P Out (LPO)

Leave P Out (LPO)，留 P 交叉验证，和 LOO 比较类似，每一轮将 P 个样本作为 验证集，剩余的 N - P 个样本作为 训练集。

LPO 的交叉验证次数为 $$C^{P}_{N}$$，并不等于 K-Flod ($$ K = \frac{N}{P} $$)，这是由于 K-Fold 各个验证集中的样本不会重复，而 LPO 会重复。   
所以，LPO 的迭代次数会随着样本数量的增加而组合增长，计算成本非常昂贵，大样本集一般不选用。

sklearn 中，可通过类 [sklearn.model_selection.LeavePOut](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePOut.html) 实现 LPO。

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.model_selection import LeavePOut

lpo = LeavePOut(p=3)

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [3, 5], [3, 5]])
y = np.array([1, 2, 3, 4, 5, 1])

for train_index, test_index in lpo.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
// TRAIN: [3 4 5] TEST: [0 1 2]
// TRAIN: [2 4 5] TEST: [0 1 3]
// TRAIN: [2 3 5] TEST: [0 1 4]
// TRAIN: [2 3 4] TEST: [0 1 5]
// TRAIN: [1 4 5] TEST: [0 2 3]
// TRAIN: [1 3 5] TEST: [0 2 4]
// TRAIN: [1 3 4] TEST: [0 2 5]
// TRAIN: [1 2 5] TEST: [0 3 4]
// TRAIN: [1 2 4] TEST: [0 3 5]
// TRAIN: [1 2 3] TEST: [0 4 5]
// TRAIN: [0 4 5] TEST: [1 2 3]
// TRAIN: [0 3 5] TEST: [1 2 4]
// TRAIN: [0 3 4] TEST: [1 2 5]
// TRAIN: [0 2 5] TEST: [1 3 4]
// TRAIN: [0 2 4] TEST: [1 3 5]
// TRAIN: [0 2 3] TEST: [1 4 5]
// TRAIN: [0 1 5] TEST: [2 3 4]
// TRAIN: [0 1 4] TEST: [2 3 5]
// TRAIN: [0 1 3] TEST: [2 4 5]
// TRAIN: [0 1 2] TEST: [3 4 5]
{%endace%}

### 1.5. Shuffle Split

Shuffle Split(Random Permutations)，首先将样本集打乱，然后根据用户指定的比例，随机的将它们分成一对训练集和验证集。

> 注意：随机分割不能保证所有 Fold 都是不同的，尤其在小数据集时很容易重复。

sklearn 中，可通过类 [sklearn.model_selection.ShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html) 实现 Shuffle Split。

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.model_selection import ShuffleSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [3, 5], [3, 5]])
y = np.array([1, 2, 3, 4, 5, 1])

ss1 = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
for train_index, test_index in ss1.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
// TRAIN: [1 3 0 4] TEST: [5 2]
// TRAIN: [4 0 2 5] TEST: [1 3]
// TRAIN: [1 2 4 0] TEST: [3 5]
// TRAIN: [3 4 1 0] TEST: [5 2]
// TRAIN: [3 5 1 0] TEST: [2 4]

ss2 = ShuffleSplit(n_splits=5, train_size=0.5, test_size=.25, random_state=0)
for train_index, test_index in ss2.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
// TRAIN: [1 3 0] TEST: [5 2]
// TRAIN: [4 0 2] TEST: [1 3]
// TRAIN: [1 2 4] TEST: [3 5]
// TRAIN: [3 4 1] TEST: [5 2]
// TRAIN: [3 5 1] TEST: [2 4]
{%endace%}

## 2. CV For Stratification Data

一些分类问题可能在目标类的分布上表现出很大的不平衡：例如在医疗数据当中得癌症的人比不得癌症的人少很多。  
在这种情况下，就需要使用分层采样，以确保在每个训练集和验证集中与总样本集近似保留相同的分类比例。

### 2.1. Stratified K-Fold

Stratified K-Fold 是 K-Fold 的变种，不过每份 Fold 中各结果分类的比例与总样本集的保持一致。

sklearn 中，可通过类 [sklearn.model_selection.StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) 实现 Stratified K-Fold。

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=2)

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [3, 5], [3, 5]])
y = np.array([0, 0, 1, 1, 1, 1])

for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
// TRAIN: [1 4 5] TEST: [0 2 3]
// TRAIN: [0 2 3] TEST: [1 4 5]
{%endace%}

### 2.2. Repeated Stratified K-Fold

Repeated Stratified K-Fold，是指重复进行多次 Stratified K-Fold。

sklearn 中，可通过类 [sklearn.model_selection.RepeatedStratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html) 实现 Repeated Stratified K-Fold。

### 2.3. Stratified Shuffle Split

Stratified Shuffle Split 是 Shuffle Split 的变种，不过每次随机验证集中各结果分类的比例与总样本集的保持一致。

sklearn 中，可通过类 [sklearn.model_selection.StratifiedShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html) 实现 Stratified Shuffle Split。

## 3. CV For Grouped Data

如果数据的生成过程依赖于样本的 group，那么样本分布就不符合 i.i.d. 的假设了。

例如，某班级的跳远成绩，每个学生均跳远多次，那么成绩样本里会带有这个学生的id，这就是 group 信息。

理论上，我们认为同一 group 的样本是比较相近的。在这种情况下，如果属于同一 group 的样本同时分布在 训练集 和 验证集，会导致 验证集 不够独立，进而影响模型验证。

因此，带有 group 的场景，同一 group 就需要在同一样本数据集：要么同时在训练集，要么同时在验证集。

### 3.1. Group K-Fold

Group K-Fold 是 K-Fold 的变种：同一 group 的样本不会出现在不同的 Fold 中，因此，group 的数量必须大于等于 Fold 的数量。

sklearn 中，可通过类 [sklearn.model_selection.GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html) 实现 Group K-Fold。

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=2)

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])
groups = np.array([0, 0, 2, 2])

for train_index, test_index in gkf.split(X, y, groups):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
// TRAIN: [0 1] TEST: [2 3]
// TRAIN: [2 3] TEST: [0 1]
{%endace%}

### 3.2. Leave One Group Out

Leave One Group Out 将 一 个 group 的样本作为 验证集。

sklearn 中，可通过类 [sklearn.model_selection.LeaveOneGroupOut](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html) 实现 Leave One Group Out。

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()

X = np.array([[1, 2], [3, 4], [5, 6], [5, 6]])
y = np.array([1, 2, 1, 3])
groups = np.array([1, 2, 3, 3])

for train_index, test_index in logo.split(X, y, groups):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
// TRAIN: [1 2 3] TEST: [0]
// TRAIN: [0 2 3] TEST: [1]
// TRAIN: [0 1] TEST: [2 3]
{%endace%}

### 3.3. Leave P Groups Out

Leave P Groups Out 将 P 个 group 的样本作为 验证集。

sklearn 中，可通过类 [sklearn.model_selection.LeavePGroupsOut](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePGroupsOut.html) 实现 Leave P Group Out。

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.model_selection import LeavePGroupsOut

lpgo = LeavePGroupsOut(n_groups=2)

X = np.array([[1, 2], [3, 4], [5, 6], [5, 6]])
y = np.array([1, 2, 1, 3])
groups = np.array([1, 2, 3, 3])

for train_index, test_index in lpgo.split(X, y, groups):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
// TRAIN: [2 3] TEST: [0 1]
// TRAIN: [1] TEST: [0 2 3]
// TRAIN: [0] TEST: [1 2 3]
{%endace%}

### 3.4. Group Shuffle Split

Group Shuffle Split，相当于 Shuffle Split 和 Leave P Groups Out 的变体。比较有用一个场景是，当 group 数量很多时，Leave P Groups Out 需要遍历所有 group，而 Group Shuffle Split 就可以指定随机的次数 n_splits。

sklearn 中，可通过类 [sklearn.model_selection.GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html) 实现 Group Shuffle Split。

{%ace edit=true, lang='java'%}
from sklearn.model_selection import GroupShuffleSplit

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 0.001]
y = ["a", "b", "b", "b", "c", "c", "c", "a"]
groups = [1, 1, 2, 2, 3, 3, 4, 4]

gss = GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0)

for train_index, test_index in gss.split(X, y, groups):
    print("TRAIN:", train_index, "TEST:", test_index)
// TRAIN: [0 1 2 3] TEST: [4 5 6 7]
// TRAIN: [2 3 6 7] TEST: [0 1 4 5]
// TRAIN: [2 3 4 5] TEST: [0 1 6 7]
// TRAIN: [4 5 6 7] TEST: [0 1 2 3]
{%endace%}

## 4. CV For Time Series Data

时间序列数据 的特点是时间相近的样本间具有一定的相关性，而传统的 CV 方法假设数据是独立同分布的，会丢失这种时间相关性。因此，引入了 Time Series Split 来处理时间序列数据。

Time Series Split 是 K-Fold 的一个变体，总体思路是 训练集 必须是 验证集 之前的元素：

1. 将第一个样本从总样本集中剔除，因为它之前已没有元素，如果进入 验证集 则 训练集 就为空；
2. 根据 n_splits 使用 K-Fold 方法将剩余样本集分成 K 份。不过需要注意，相邻样本必须在一起，例如将样本集 {6,7,8,9} 分成 2 份，就只能分成 {6,7},{8,9}；
3. 循环将 K 个 Fold 作为 验证集，然后将此验证集之前的元素作为 训练集；

sklearn 中，可通过类 [sklearn.model_selection.TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) 实现 Time Series Split。

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])

for i in np.arange(2, 5):
    print("n_splits =", i, ": ")
    for train_index, test_index in TimeSeriesSplit(n_splits=i).split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
// n_splits = 2 :
// TRAIN: [0 1] TEST: [2 3]
// TRAIN: [0 1 2 3] TEST: [4 5]
// n_splits = 3 :
// TRAIN: [0 1 2] TEST: [3]
// TRAIN: [0 1 2 3] TEST: [4]
// TRAIN: [0 1 2 3 4] TEST: [5]
// n_splits = 4 :
// TRAIN: [0 1] TEST: [2]
// TRAIN: [0 1 2] TEST: [3]
// TRAIN: [0 1 2 3] TEST: [4]
// TRAIN: [0 1 2 3 4] TEST: [5]
{%endace%}

上例中，样本总数量为 6。由于验证集不能包含第1个样本，故剩余的可作为验证集的样本数量为 5。为最大限度的提升训练集中的信息量，会优先从后往前取样本进入验证集。所以，当 n_splits=3 时会取后面 3 个元素，当 n_splits=2 时会取后面 4 个元素。

## 5. Predefined Split

sklearn 中，我们可通过类 [sklearn.model_selection.PredefinedSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.PredefinedSplit.html) 自定义验证集。

> test_fold中，-1 会被忽略，其余值会分别作为一个验证集，循环遍历。如下例中，就会交叉验证 3 次。

{%ace edit=true, lang='java'%}
import numpy as np
from sklearn.model_selection import PredefinedSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1, 1])
test_fold = [0, 1, -1, 2, 0]

ps = PredefinedSplit(test_fold)
print("n_splits:", ps.get_n_splits())
// n_splits: 3

for train_index, test_index in ps.split():
    print("TRAIN:", train_index, "TEST:", test_index)
// TRAIN: [1 2 3] TEST: [0 4]
// TRAIN: [0 2 3 4] TEST: [1]
// TRAIN: [0 1 2 4] TEST: [3]
{%endace%}


