<!-- toc -->

# Pipelines And Composite Estimators

转换器(Transformers) 经常与 classifiers、regressors 等 Estimator 组合，形成一个 组合Estimator。这个组合通常是通过 **Pipeline** 实现的。

## 1. Pipeline

[Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) 将多个 Estimator 串连为一个序列，例如分类模型中的 特征选择、正则化处理、分类器 的处理序列。在这些序列中，除最后一个外，都需要包含 transform 函数。

{%ace edit=false, lang='java'%}
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

estimators = [('reduce_dim', PCA()), ('clf', SVC())]
pipe = Pipeline(estimators)
print(pipe)
// Pipeline(memory=None,
//          steps=[('reduce_dim',
//                  PCA(copy=True, iterated_power='auto', n_components=None,
//                      random_state=None, svd_solver='auto', tol=0.0,
//                      whiten=False)),
//                 ('clf',
//                  SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
//                      decision_function_shape='ovr', degree=3,
//                      gamma='auto_deprecated', kernel='rbf', max_iter=-1,
//                      probability=False, random_state=None, shrinking=True,
//                      tol=0.001, verbose=False))],
//          verbose=False)
{%endace%}

在 Pipeline 中，可以统一设置所有 Estimator 可选的参数范围，再通过 [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 调参。

考虑到 GridSearchCV 时，在每个 Estimator 中需要对同样的输入数据和参数进行多次计算。为提升性能，Pipeline 还提供了对于各 Estimator 的 fit 结果的缓存功能。

{%ace edit=false, lang='java'%}
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

estimators = [('reduce_dim', PCA()), ('clf', SVC())]
cachedir = mkdtemp()
pipe = Pipeline(estimators, memory=cachedir)
print(pipe)
{%endace%}

## 2. Transforming Target In Regression

[TransformedTargetRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html) 在回归模型 **fit** 之前对目标值 **y** 进行转化，常用于回归问题中的非线性变换。

{%ace edit=false, lang='java'%}
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
tt = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log, inverse_func=np.exp)
X = np.arange(4).reshape(-1, 1)
y = np.exp(2 * X).ravel()
tt.fit(X, y) 
print(tt.score(X, y))
// 1.0
print(tt.regressor_.coef_)
// [2.]
{%endace%}

## 3. FeatureUnion

Pipeline 是将多个 Estimator 串连为一个序列，是串行计算。而 [FeatureUnion](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html) 是将多个 Estimator 对同一份输入并行计算，最终将多份并行计算结果中的特征合并到一起。

> 例如，Estimator\_A 在 fit 后输出的特征维度为1，Estimator\_B 在 fit 后输出的特征维度为2，Estimator_C 在 fit 后输出的特征维度为5，那么三者顺序 Pipeline 后输出的结果为 5 维，而 FeatureUnion 后输出的结果为 1+2+5=8 维。

{%ace edit=false, lang='java'%}
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

estimators = [('linear_pca', PCA()), ('kernel_pca', KernelPCA())]
combined = FeatureUnion(estimators)
print(combined)
{%endace%}

