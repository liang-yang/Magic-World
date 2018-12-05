<!-- toc -->

# SGDClassifier

---

**class** [sklearn.linear_model.SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier)

SGDClassifier 是通过 随机梯度下降 训练线性分类器，如 SVM、逻辑回归等。

SGDClassifier 可以通过 partial_fit 函数支持 Mini-Batch。

SGDClassifier 在 稀疏 数据也有较好的表现。

## 1. Parameters

Parameter | Default | Comment | Note
:-:|:-:|:-:|:-:
loss | hinge | 损失函数，hinge、log、perceptron等 | 默认的hinge表示SVM
penalty | l2 | regularization term，正则化惩罚项 | 默认的l2适用于SVM，l1、elasticnet 更适用于 稀疏数据
alpha | 0.0001 | penalty 的系数 | -
l1_ratio | 0.15 |  | -
fit_intercept | True | 是否需要评估 intercept（截距项） | -
max_iter | None | epochs，迭代的最大轮数 | 仅在 fit 方法生效
tol | None | 停止标准：loss > previous_loss - tol | -
shuffle | True | 每轮迭代是否需要重新打散训练数据 | -
verbose | 0 |  | -
epsilon | 0.1 |  | - 
n_jobs | None | 计算使用的CPU数量 | -
random_state | None | shuffle 数据时使用的随机种子 | -
learning_rate | 'optimal' | 学习率。'constant': $$ \eta = \eta_0$$； 'optimal': $$\eta = \frac{1.0}{\alpha \times (t + t_0)}$$； 'invscaling': $$\eta = \frac{\eta_0}{t^{power\_t}}$$；'adaptive': | -
eta0 | 0.0 | $$\eta_0$$, 'constant','invscaling','adaptive'模式下的初始学习率  | 对默认模型 'optimal' 无效
power_t | 0.5 | 'invscaling'模式下的幂参数 | -
early_stopping | False |  | -
validation_fraction | 0.1 | early_stopping 时，从训练数据中划分出来的比例 | 仅 early_stopping 为 True 时生效
n_iter_no_change | 5 | early_stopping 时，没有明显增长的检测轮数 | 仅 early_stopping 为 True 时生效 
class_weight | None |  | -
warm_start | False |  | -
average | False |  | -
n_iter | None | 迭代的轮数 | 0.21版本将会删除









