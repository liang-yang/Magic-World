<!-- toc -->

# Perceptron

---

**class** [sklearn.linear_model.Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron)

## 1. Parameters

Parameter | Default | Comment | Note
:-:|:-:|:-:|:-:
penalty | None | regularization term，惩罚项 | 可选 'l2'、'l1'、'elasticnet' 
alpha | 0.0001 | penalty 的系数 | - 
fit_intercept | True | 是否需要评估 intercept（截距项） | - 
max_iter | 1000 | epochs，迭代的轮数 | 仅在 fit 方法生效
tol | 1e-3 | 停止标准：loss > previous_loss - tol | - 
shuffle | True | 每轮迭代是否需要重新打散训练数据 | - 
verbose |  |  | - 
eta0 | 1 |  | - 
n_jobs | None | 计算使用的CPU数量 | - 
random_state | None | shuffle 数据时使用的随机种子 | - 
early_stopping | False |  | - 
validation_fraction | 0.1 | early_stopping 时，从训练数据中划分出来的比例 | 仅 early_stopping 为 True 时生效 
n_iter_no_change | 5 | early_stopping 时，没有明显增长的检测轮数 | 仅 early_stopping 为 True 时生效 
class_weight |  |  | - 
warm_start |  |  | - 
n_iter | None | 迭代的轮数 | 0.21版本将会删除


## 2. Attributes

Parameter | Comment | Tips
:-:|:-:|:-:
coef_ | 生成的样本集 | -
intercept_ | 生成的样本集 | -
n\_iter\_ | 生成的样本集 | -

## 3. Methods





