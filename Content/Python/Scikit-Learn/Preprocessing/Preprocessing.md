<!-- toc -->

# Preprocessing

---

[sklearn.preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)





[OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)




## 1. StandardScaler

**class** [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)

标准化特征值 就是特征值减去均值（中心化，Centering）再除以标准差（缩放，Scaling）。

标准化特征值 对 每项特征 独立进行。

### 1.1. Parameters

Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
copy | boolean | True | 是否返回原始数据缩放后的副本，不会修改原始数据 | -
with_mean | boolean | True | 在缩放前是否中心化数据 | 稀疏数据尽量不要中心化，因为可能使得稀疏矩阵变成稠密矩阵
with_std | boolean | True | 是否通过除以标准差的方式缩放 | -

### 1.2. Attributes

Attributes | Data-Type | Comment | Note
:-:|:-:|:-:|:-:
scale_ | None, array[n_features] | 各项特征的缩放比例 | 通过 np.sqrt(var_) 计算，如果 with_std=False 则返回 None
mean_ | None, array[n_features] | 各项特征的均值 | 如果 with_mean=False 则返回 None
var_ | None, array[n_features] | 各项特征的方差 | 如果 with_std=False 则返回 None
n\_samples\_seen_ | int, array[n_features] | 各项特征计算使用的样本数量 | 如果特征值没有 NaNs，为 int，否则为 array

> 特征值为 NaNs 表示 missing，在 fit 时会忽略，transform 时会计算。

### 1.3. Methods

#### 1.3.1. fit

计算 均值 和 标准差 供后续使用

Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
X | array[n_samples,n_features] | - | 训练集 | -

#### 1.3.2. fit_transform

fit 后 transform

Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
X | array[n_samples,n_features] | - | 原始训练集 | -
y | array[n_samples] | None | 目标值 | -

Returns | Data-Type | Comment | Note
:-:|:-:|:-:|:-:
X_new | array[n_samples,n_features_new] | transform后新的训练集 | -
#### 1.3.3. get_params

获取参数集合

Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
deep | boolean | True | 如果True，会返回子对象 | -

Returns | Data-Type | Comment | Note
:-:|:-:|:-:|:-:
params | mapping | 参数名对应的map | -
#### 1.3.4. inverse_transform

逆 transform，即将 transform 后的训练集转换回 原始训练集

Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
X | array[n_samples,n_features] | - | 被 transform 后的训练集 | -
copy | boolean | None | 是否拷贝输入的 X | -

Returns | Data-Type | Comment | Note
:-:|:-:|:-:|:-:
X_tr | array[n_samples,n_features] | 逆 transform 后的训练集 | -

#### 1.3.5. partial_fit

当样本集 数量特别大 或者 属于流式数据 时，我们可以通过 partial_fit 实现增量fit，也叫 online learning。 与 fit 相比，每一次的 partial_fit 都会结合之前的数据，而 fit 都是全新的计算。

Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
X | array[n_samples,n_features] | - |原始训练集 | -

#### 1.3.6. set_params

设置参数，以便于人工调整

Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
params | mapping | - | 参数名对应的map | -

#### 1.3.7. transform

标准化处理：减去均值，除以标准差

Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
X | array[n_samples,n_features] | - | 原始训练集 | -
copy | boolean | None | 是否拷贝输入的 X | -

### 1.4. Examples

{%ace edit=true, lang='python'%}
from sklearn.preprocessing import StandardScaler

data = [[0., 0.], [0., 0.], [1., 1.], [1., 1.]]

scaler = StandardScaler()
scaler.fit(data)
print("scale_:",scaler.scale_)
print("mean_:",scaler.mean_)
print("var_:",scaler.var_)
print("n_samples_seen_:",scaler.n_samples_seen_)

print(scaler.transform(data))
print(scaler.transform([[2, 2]]))
{%endace%}




















