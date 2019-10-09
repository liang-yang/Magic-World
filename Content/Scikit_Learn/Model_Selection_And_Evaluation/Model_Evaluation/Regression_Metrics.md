<!-- toc -->

# Regression Metrics

---

see [Regression Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)

回归，本意为 “向平均回归” ，是高尔顿在研究豌豆的 “回归效应” 时提出的。  
回归分析，是分析 自变量 与 因变量 之间的相关关系。

> 注意：相关关系 不一定意味着 因果关系。

## 1. R2 Score

回归分析，是通过拟合回归模型，根据 自变量 预测 因变量：

- 如果 因变量预测值 与 因变量真实值 完全一致，则认为回归模型能很好地解释因变量的变化；
- 如果 因变量预测值 与 因变量真实值 偏差不大，则认为回归模型能较好地解释因变量的变化；
- 如果 因变量预测值 与 因变量真实值 偏差很大，则认为回归模型不能较好地解释因变量的变化；

而度量 回归模型 对 因变量变化 的解释程度的指标，就是 $$R^2$$。

### 1.1. SST

因变量变化 的程度，可视作其偏离均值的程度（均值被视作因变量无变化的值），通过 $$SST$$（Total Squares Sum，总平方和）度量：

$$
\bar{y} = \frac{1}{N} \cdot \sum_{i=1}^{N} y_i
$$

$$
SST = \sum_{i=1}^{N} (y_i-\overline{y})^2
$$

### 1.2. SSR 

回归模型 预测的 自变量 对 因变量 的影响，可通过 $$SSR$$（Regression Squares Sum，回归平方和）度量：

$$
SSR = \sum_{i=1}^{N} (\hat{y_i}-\overline{y})^2
$$

因此，$$SSR$$ 就是回归模型解释的因变量与自变量的变化关系。

### 1.3. SSE

回归模型的 预测因变量 与 实际因变量 之间的差异，可通过 $$SSE$$（Error Squares Sum，残差平方和）度量：

$$
SSE = \sum_{i=1}^{N} (y_i-\hat{y_i})^2
$$

因此，$$SSE$$ 就是回归模型无法解释的因变量与自变量的变化关系。

### 1.4. R2 Score

很明显，$$SSR$$ 与 $$SST$$ 越接近，说明回归模型越能解释因变量与自变量的变化关系。

因此，通过 $$SSR$$ 与 $$SST$$ 的比值来度量解释程度，这就是 $$R^2$$：

$$
R^2 = \frac{SSR}{SST}
$$

### 1.5. Adjusted R2 Score

但是，$$R^2$$ 存在一个缺陷，即随着自变量维度的增长，即自变量个数的增多（非样本个数的增加），总是会导致 $$R^2$$ 的上升，无论增加的自变量与因变量是否存在线性关系。这是由于：
    
1. 当增加自变量个数时，因变量真实值 不会改变，均值也不会变，所以 $$SST$$ 不会变；
2. 当增加自变量个数时，就给了拿出新的最优解的可能，使得 $$SSR$$ 更接近 $$SST$$。即使在最极端的情况下，也可以把新增自变量的系数设为0，也与没增加自变量个数持平。因此，增加自变量个数，肯定会使得 $$R^2$$ 更接近 1；

因此，为了避免单纯通过增加自变量个数的方式来提升 $$R^2$$ ，提出了修正后的 $$R^2$$（Adjusted $$R^2$$）：

$$
R_a^2 = 1-(1-R^2)(\frac{n-1}{n-k-1})
$$

通过公式可以看出，修正后的 $$R^2$$ 比 $$R^2$$ 要小。







，$$SSE$$ 越小越好







### 1.5. 


回归模型成立的最基础的假设条件是：因变量 随 自变量 的变化而变化。但是，拟合出的回归模型能够解释而 $$R^2$$ 就是用于分析 因变量 随 自变量 变化而变化的程度。











---


$$
R^2(y,\hat{y}) = 1 - \frac{\sum^{N}_{i=1}(y_i-\hat{y}_i)^2}{\sum^{N}_{i=1}(y_i-\bar{y})^2}
$$

需要认识到，
1. 如果回归模型完全成立，那么因变量y的所有变化都可以由x解释。
2. 如果回归模型仅部分成立，那么x仅可解释y的部分变化。
3. 极端情况下y与x的回归模型完全不成立，那么随着x的变化y无变化。
    
这样，就通过观察x对y的变化的解释程度，来衡量回归模型对测量数据的拟合程度。
    
y的所有变化（偏离均值），可以通过 SST 标识：
$$
SST = \sum(y_i-\overline{y})^2
$$
x的变化对y的变化的影响，可以通过 SSR 标识：
$$
SSR = \sum(\hat{y_i}-\overline{y})^2
$$
x无法解释的y的变化，可以通过 SSE 标识：
$$
SSE = \sum(y_i-\hat{y_i})^2
$$
    
可以证明：
$$
SST = SSR + SSE
$$
> 证明可 [参考](https://www.zybang.com/question/ba863fb4d7d87ad2871a3d16f7cf4b7c.html)。 通过证明过程可以看出，一定要在 最小二乘法 的原则下，才有 `$SST = SSR + SSE$`。
    
因此，回归模型拟合的好坏，可通过 SSR 占 SST 的比例来衡量，即：
$$
\text{判定系数}R^2=SSR/SST
$$






## 1. Explained Variance Score

$$
Explained \  Variance(y, \hat{y}) = 1 - \frac{Var\{y - \hat{y}\}}{Var\{y\}}
$$



## 2. Max Error

$$
Max \  Error(y, \hat{y}) = \max(|y_i - \hat{y}_i|) 
$$



## 3. Mean Absolute Error(MAE)

$$
MAE(y, \hat{y}) = \frac{1}{N} \cdot \sum_{i=1}^{N} (|y_i - \hat{y}_i|)
$$


## 4. Mean Squared Error(MSE)

$$
MSE(y, \hat{y}) = \frac{1}{N} \cdot \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

## 5. Mean Squared Logarithmic Error(MSLE)

$$
MSLE(y, \hat{y}) = \frac{1}{N} \cdot \sum_{i=1}^{N} (\ln(1+y_i)-\ln(1+\hat{y}_i))^2
$$

## 6. Median Absolute Error(MedAE)

$$
MedAE(y, \hat{y}) = median(|y_1 - \hat{y}_1|,\cdots,|y_N - \hat{y}_N|) 
$$


