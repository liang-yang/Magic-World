<!-- toc -->

# Tuning Hyper-Parameters

---

see [Tuning Hyper-Parameters](https://scikit-learn.org/stable/modules/grid_search.html)

> Tuning Hyper-Parameters 使用的类及函数均在 [sklearn.model_selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection) 包中。

机器学习，就是 样本数据 通过拟合 算法模型 得到 模型参数 的过程。而除了 模型参数 以外，还有一些参数是无法通过拟合得到的，这就是 **超参数**。

一般来说，**超参数** 是基于应用场景、根据经验值人为设定，并作为入参传递给算法模型，如 SVC 中的 C、kernel、gamma，Lasso 中的 alpha 等。 但是，既然 超参数 是人为设置的，那怎么判断设置的好坏呢？机器学习中，一般使用 **Grid Search**（网格搜索）的方案：

- 在所有候选的超参数选择中，通过循环遍历，尝试每一种可能性，表现最好的超参数就是最终的结果；
- 为什么叫网格搜索？以有两个超参数的模型为例，超参数a有3种可能，超参数b有4种可能，把所有可能性列出来，可以表示成一个3*4的表格，其中每个cell就是一个网格，循环过程就像是在每个网格里遍历、搜索；

实际操作中，一般是将 Gird Search 和 Cross Validation 放到一起计算，这是为了在网格搜索中更多的使用训练集的特征数据。

## 1. Exhaustive Grid Search

sklearn 中，可通过 [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 进行网格搜索。

{%ace edit=true, lang='java'%}
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pandas as pd

iris = datasets.load_iris()
param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]
grid_search = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=5)
grid_search.fit(iris.data, iris.target)
cv_results = pd.DataFrame(grid_search.cv_results_)

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 9999999)
cols = ['params', 'param_C', 'param_kernel', 'param_gamma', 'mean_fit_time', 'std_fit_time',
        'mean_score_time', 'std_score_time', 'split0_test_score', 'split1_test_score',
        'split2_test_score', 'split3_test_score', 'split4_test_score', 'mean_test_score', 'std_test_score',
        'rank_test_score']
print(cv_results.loc[:, cols])
//                                            params param_C param_kernel param_gamma  mean_fit_time  std_fit_time  mean_score_time  std_score_time  split0_test_score  split1_test_score  split2_test_score  split3_test_score  split4_test_score  mean_test_score  std_test_score  rank_test_score
// 0                    {'C': 1, 'kernel': 'linear'}       1       linear         NaN       0.001657      0.001607         0.000609        0.000219           0.966667           1.000000           0.966667           0.966667           1.000000         0.980000        0.016330                1
// 1                   {'C': 10, 'kernel': 'linear'}      10       linear         NaN       0.000916      0.000117         0.000572        0.000143           1.000000           1.000000           0.900000           0.966667           1.000000         0.973333        0.038873                5
// 2                  {'C': 100, 'kernel': 'linear'}     100       linear         NaN       0.001309      0.000137         0.000733        0.000085           1.000000           1.000000           0.900000           0.933333           1.000000         0.966667        0.042164                6
// 3                 {'C': 1000, 'kernel': 'linear'}    1000       linear         NaN       0.001248      0.000343         0.000482        0.000105           1.000000           1.000000           0.900000           0.933333           1.000000         0.966667        0.042164                6
// 4       {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}       1          rbf       0.001       0.001869      0.000190         0.000823        0.000099           0.866667           0.966667           0.833333           0.966667           0.933333         0.913333        0.054160               10
// 5      {'C': 1, 'gamma': 0.0001, 'kernel': 'rbf'}       1          rbf      0.0001       0.003754      0.002587         0.000791        0.000060           0.866667           0.966667           0.833333           0.966667           0.933333         0.913333        0.054160               10
// 6      {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}      10          rbf       0.001       0.003308      0.003564         0.000901        0.000186           0.900000           0.966667           0.866667           0.933333           1.000000         0.933333        0.047140                8
// 7     {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}      10          rbf      0.0001       0.005135      0.002691         0.001014        0.000256           0.866667           0.966667           0.833333           0.966667           0.933333         0.913333        0.054160               10
// 8     {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}     100          rbf       0.001       0.000843      0.000091         0.000610        0.000021           0.966667           1.000000           0.966667           0.966667           1.000000         0.980000        0.016330                1
// 9    {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'}     100          rbf      0.0001       0.000993      0.000182         0.000568        0.000207           0.900000           0.966667           0.866667           0.933333           1.000000         0.933333        0.047140                8
// 10   {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}    1000          rbf       0.001       0.000744      0.000228         0.000482        0.000128           0.966667           1.000000           0.966667           0.966667           1.000000         0.980000        0.016330                1
// 11  {'C': 1000, 'gamma': 0.0001, 'kernel': 'rbf'}    1000          rbf      0.0001       0.000857      0.000119         0.000594        0.000071           0.966667           1.000000           0.966667           0.966667           1.000000         0.980000        0.016330                1

print(grid_search.best_estimator_)
// SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
//     decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
//     kernel='linear', max_iter=-1, probability=False, random_state=None,
//     shrinking=True, tol=0.001, verbose=False)
print(grid_search.best_score_)
// 0.98
print(grid_search.best_params_)
// {'C': 1, 'kernel': 'linear'}
print(grid_search.best_index_)
// 0
{%endace%}

### 1.1. cv_results

cv\_results\ 中主要包含 三 类结果：

1. params 类。这类结果在上例中的具体参数为 ['params', 'param_C', 'param_kernel', 'param_gamma']，主要记录当次遍历使用的超参数组合，即我们的“网格”；
2. time 类。这类结果在上例中的具体参数为 ['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time']，主要记录fit 和 score 执行的时间；
3. score 类。这类结果在上例中的具体参数为 ['split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'mean_test_score', 'std_test_score', 'rank_test_score']。其中，['split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score'] 表示5次cv的验证集的得分，['mean_test_score', 'std_test_score'] 表示这些得分的均值和标准差，'rank_test_score' 表示得分均值的排名（在  GridSearchCV 的构造函数中设置入参 return\_train\_score=True，则会返回训练集的得分）；

### 1.2. refit

GridSearchCV 的 refit 参数，表示在确定 超参数组合 后是否重新拟合模型参数。我们考虑网格计算的完整步骤：

1. 遍历所有的 超参数 组合，即 “网格”；
2. 交叉验证所有的 cv，针对每个测试Fold得到一个score，最终得到一个score均值（注意，每个测试Fold拟合的模型参数不一定一致）；
3. 经过以上遍历过程，取score均值最高的超参数组合作为 超参数最优组合；
4. 以 超参数最优组合 作为入参，重新 fit 所有的样本训练集，得到最终模型参数；

refit 为 True 时，才会执行上面的第 4 个步骤，也才会有 ['best\_estimator\_'、'best\_score\_'、'best\_params\_'、'best\_index\_'] 的输出，分别表示得分最高的 estimator（包含超参数组合且已拟合样本集得到模型参数），其最终得分，以及 超参数组合 和 所在index位置。

## 2. Randomized Parameter Optimization

由于 GridSearchCV 需要遍历所有的 超参数组合，这样会存在两个问题：

1. 当超参数组合特别多时，计算会特别耗时；
2. 当超参数取值非 离散型列表 而是 连续型分布 时，GridSearchCV 就无法处理。

因此，引入了 Randomized Parameter Optimization（随机参数优化）。它会从 某项超参数的取值范围（离散型列表 or 连续型分布） 中，随机抽取超参数值，与其他项的超参数值组成超参数组合，以控制计算量。

> 随机参数优化 的 有效性 可参考 [随机搜索RandomizedSearchCV原理](https://blog.csdn.net/qq_36810398/article/details/86699842)

sklearn 中，可通过 [sklearn.model_selection.RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) 实现随机参数优化。相对于 GridSearchCV，它有两点变化：

1. 将参数 param\_grid 修改为了 param\_distributions，支持指定连续分布；
2. 新增参数 n_iter，用以指定所取的 超参数组合 的数量；

{%ace edit=true, lang='java'%}
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from scipy.stats import randint

iris = datasets.load_iris()
param_distributions = {'C': randint(1, 11), 'gamma': [0.001, 0.0001], 'kernel': ['rbf', 'linear']}

grid_search = RandomizedSearchCV(svm.SVC(), param_distributions=param_distributions, cv=5, n_iter=20)
grid_search.fit(iris.data, iris.target)
cv_results = pd.DataFrame(grid_search.cv_results_)

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 9999999)
cols = ['params', 'param_C', 'param_kernel', 'param_gamma', 'mean_fit_time', 'std_fit_time',
        'mean_score_time', 'std_score_time', 'split0_test_score', 'split1_test_score',
        'split2_test_score', 'split3_test_score', 'split4_test_score', 'mean_test_score', 'std_test_score',
        'rank_test_score']
print(cv_results.loc[:, cols])
//                                             params param_C param_kernel param_gamma  mean_fit_time  std_fit_time  mean_score_time  std_score_time  split0_test_score  split1_test_score  split2_test_score  split3_test_score  split4_test_score  mean_test_score  std_test_score  rank_test_score
// 0        {'C': 5, 'gamma': 0.001, 'kernel': 'rbf'}       5          rbf       0.001       0.001487      0.001168         0.000610        0.000173           0.900000           0.966667           0.866667           0.933333           0.900000         0.913333        0.033993               11
// 1       {'C': 6, 'gamma': 0.0001, 'kernel': 'rbf'}       6          rbf      0.0001       0.001511      0.000206         0.000604        0.000062           0.866667           0.966667           0.833333           0.966667           0.933333         0.913333        0.054160               11
// 2       {'C': 5, 'gamma': 0.0001, 'kernel': 'rbf'}       5          rbf      0.0001       0.001660      0.000547         0.000710        0.000171           0.866667           0.966667           0.833333           0.966667           0.933333         0.913333        0.054160               11
// 3       {'C': 3, 'gamma': 0.0001, 'kernel': 'rbf'}       3          rbf      0.0001       0.002026      0.000180         0.000837        0.000036           0.866667           0.966667           0.833333           0.966667           0.933333         0.913333        0.054160               11
// 4       {'C': 2, 'gamma': 0.0001, 'kernel': 'rbf'}       2          rbf      0.0001       0.001375      0.000341         0.000542        0.000125           0.866667           0.966667           0.833333           0.966667           0.933333         0.913333        0.054160               11
// 5    {'C': 9, 'gamma': 0.0001, 'kernel': 'linear'}       9       linear      0.0001       0.000942      0.000282         0.000534        0.000038           1.000000           1.000000           0.900000           0.966667           1.000000         0.973333        0.038873                4
// 6    {'C': 4, 'gamma': 0.0001, 'kernel': 'linear'}       4       linear      0.0001       0.000602      0.000159         0.000402        0.000087           0.966667           1.000000           0.933333           0.966667           1.000000         0.973333        0.024944                4
// 7    {'C': 8, 'gamma': 0.0001, 'kernel': 'linear'}       8       linear      0.0001       0.000966      0.000341         0.000643        0.000150           1.000000           1.000000           0.900000           0.966667           1.000000         0.973333        0.038873                4
// 8    {'C': 2, 'gamma': 0.0001, 'kernel': 'linear'}       2       linear      0.0001       0.000766      0.000183         0.000508        0.000151           0.966667           1.000000           0.966667           0.966667           1.000000         0.980000        0.016330                1
// 9        {'C': 5, 'gamma': 0.001, 'kernel': 'rbf'}       5          rbf       0.001       0.001315      0.000138         0.000707        0.000065           0.900000           0.966667           0.866667           0.933333           0.900000         0.913333        0.033993               11
// 10    {'C': 5, 'gamma': 0.001, 'kernel': 'linear'}       5       linear       0.001       0.001067      0.000240         0.000809        0.000160           1.000000           1.000000           0.933333           0.966667           1.000000         0.980000        0.026667                1
// 11       {'C': 6, 'gamma': 0.001, 'kernel': 'rbf'}       6          rbf       0.001       0.001576      0.001389         0.000491        0.000223           0.900000           0.933333           0.833333           0.933333           0.900000         0.900000        0.036515               20
// 12       {'C': 5, 'gamma': 0.001, 'kernel': 'rbf'}       5          rbf       0.001       0.000865      0.000119         0.000423        0.000077           0.900000           0.966667           0.866667           0.933333           0.900000         0.913333        0.033993               11
// 13   {'C': 10, 'gamma': 0.001, 'kernel': 'linear'}      10       linear       0.001       0.000514      0.000044         0.000369        0.000088           1.000000           1.000000           0.900000           0.966667           1.000000         0.973333        0.038873                4
// 14    {'C': 6, 'gamma': 0.001, 'kernel': 'linear'}       6       linear       0.001       0.000897      0.000365         0.000585        0.000198           1.000000           1.000000           0.900000           0.966667           1.000000         0.973333        0.038873                4
// 15    {'C': 7, 'gamma': 0.001, 'kernel': 'linear'}       7       linear       0.001       0.000601      0.000089         0.000367        0.000035           1.000000           1.000000           0.900000           0.966667           1.000000         0.973333        0.038873                4
// 16      {'C': 8, 'gamma': 0.0001, 'kernel': 'rbf'}       8          rbf      0.0001       0.001178      0.000314         0.000449        0.000045           0.866667           0.966667           0.833333           0.966667           0.933333         0.913333        0.054160               11
// 17  {'C': 10, 'gamma': 0.0001, 'kernel': 'linear'}      10       linear      0.0001       0.000658      0.000137         0.000493        0.000140           1.000000           1.000000           0.900000           0.966667           1.000000         0.973333        0.038873                4
// 18   {'C': 2, 'gamma': 0.0001, 'kernel': 'linear'}       2       linear      0.0001       0.001003      0.000091         0.000638        0.000068           0.966667           1.000000           0.966667           0.966667           1.000000         0.980000        0.016330                1
// 19      {'C': 3, 'gamma': 0.0001, 'kernel': 'rbf'}       3          rbf      0.0001       0.001778      0.000332         0.000877        0.000316           0.866667           0.966667           0.833333           0.966667           0.933333         0.913333        0.054160               11
{%endace%}


