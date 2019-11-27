<!-- toc -->

# Loss Function

损失函数，也叫代价函数（Cost Function），用作评价机器学习模型的预测值和真实值差异的程度：一般来说，损失函数越小，模型越好。 

因此，机器学习的过程，往往转化为求解 损失函数 最小值的过程。

符号 | 含义
:-: | :-:
$$S^n$$ | 样本集，包含 $$n$$ 个样本
$$x_i$$ | 第 $$i$$ 个样本的特征向量
$$y_i$$ | 第 $$i$$ 个样本的真实结果
$$f(x_i)$$ | 第 $$i$$ 个样本的预测结果
$$L_i = L(y_i, f(x_i))$$ | 第 $$i$$ 个样本的损失函数，简写为 $$L_i$$
$$L$$ | 样本集的损失函数

## 1. Classification

分类模型的损失函数，可分为 概率模型 和 非概率模型：

- 概率模型：一般是基于概率分布的理论，利用 极大似然估计 方法，目标是使得真实结果概率最大化的分布；
- 非概率模型：一般是基于预测结果与真实结果的差异定义损失函数，目标是将损失函数最小化；

非概率模型 一般也存在 概率解释。

> 1. $$P(y_i|x_i)$$ 表示第 $$i$$ 个样本预测正确的概率；
> 2. 为便于描述，下文以二分类为例进行分析：$$y_i \in \{0,1\}$$，1 代表正分类，0 代表负分类；
> 3. 扩展到多分类的场景，差别就在于预测结果 $$f(x_i)$$ 由正分类概率的标量转化为各分类概率的向量，同时 $$P(y_i|x_i)$$ 的计算表达式也相应改变，但整体分析逻辑是不变的；

损失函数 | $$L_i$$ | $$P(y_i|x_i)$$ | 场景
:-: | :-: | :-: | :-:
Zero One | $$\displaystyle \begin{cases} 0 &\text{if } f(x_i) = y_i \\ 1 &\text{if } f(x_i) \ne y_i \end{cases}$$ | - | 分类$$f(x) \in \{0,1\}$$。 0-1损失函数逻辑上最直接，但数学性质不好，变化不平滑，难以学习。
Mean Squared Error | $$\displaystyle \frac{1}{2} (y_i-f(x_i))^2 $$ | $$\displaystyle \frac{1}{\sqrt{2\pi}s} e^{\frac{-(y_i-f(x_i))^2}{2s^2}}$$ | 正分类概率$$f(x) \in [0,1]$$。 MSE 相对 Zero One 更为平滑，数学性质更好，相对更为常用。 但是，MSE 在 $$f(x) \to 0/1$$ 时梯度趋近于0，导致训练很慢。
Cross Entropy | $$\displaystyle - (y_i \ln f(x_i) + (1-y_i) \ln (1-f(x_i)))$$ | $$\displaystyle f(x_i)^{y_i}(1-f(x_i))^{1-y_i}$$ | 正分类概率$$f(x) \in [0,1]$$。 交叉熵可以完美解决MSE梯度下降过慢的问题，具有 “误差越大，梯度下降越快” 的良好性质。
Log | $$\displaystyle - \ln \Big(y_if(x_i) + (1-y_i)(1-f(x_i)\Big) $$ | $$\displaystyle y_if(x_i) + (1-y_i)(1-f(x_i)$$ | 正分类概率$$f(x) \in [0,1]$$。 对数损失函数常用于逻辑回归问题中。

样本集的损失函数 $$L$$ 是 $$L_i$$ 的均值：

$$
\displaystyle L = \frac{1}{n} \sum^n_{i=1} L_i
$$

### 1.1. Mean Squared Error

$$
\displaystyle L_i = \frac{1}{2} (y_i-f(x_i))^2
$$

- **梯度弥散**

    对于分类模型，当 $$f(x) \to 0$$ 或 $$f(x) \to 1$$ 时，有 $$x \to \infty, f^{'}(x) \to 0$$。 
        
    > 分类模型，一般会通过类似 Sigmoid 的转换函数将数值转化到 [0,1] 的概率区间。    
    假设转换函数为 $$\sigma(x)$$，其一般存在一个很重要的性质：当 $$x \to \infty$$ 时，有 $$\sigma(x) \to 0/1, \sigma^{'}(x) \to 0$$。    
    这是因为转换函数平滑的将无限的定义域映射到有限的值域，如果不具备此项性质就会越界。
        
    此时，MSE 的梯度 $$\displaystyle \frac{\partial L_i}{\partial f(x_i)} = -(y_i-f(x_i))f^{'}(x_i)$$ 也趋近于0，导致训练很慢。 
        
    需要注意，MSE损失函数的饱和区间仅与预测值大小有关，与预测是否准确无关。而我们理想的场景是 “误差越大，梯度下降越快”。由此，引入了 Cross Entropy 损失函数。
    
- **概率解释**

    通过 极大似然估计 方法，MSE 也存在概率解释。 **关键点在于假设拟合函数与真实值之间的差异源于服从高斯分布的误差。**不妨设误差的方差为 s，则取预测值的概率为：
    
    $$
    P(y_i|x_i) = \frac{1}{\sqrt{2\pi}s} e^{\frac{-(y_i-f(x_i))^2}{2s^2}}
    $$

    取对数，则有 
    
    $$
    \ln P(y_i|x_i) = \ln \frac{1}{\sqrt{2\pi}s} + \frac{-(y_i-f(x_i))^2}{2s^2}
    $$
    
    剔除常量，同时添加负号，将 极大概率 转换为 极小损失，最终得到：
    
    $$
    L_i = (y_i-f(x_i))^2
    $$

    需要说明，我们在推导中并没有要求 $$y_i$$ 和 $$f(x_i)$$ 是概率值，而是在第一个公式处将其转换为了概率。因此，MSE也适用于回归场景。

### 1.2. Cross Entropy

$$
\displaystyle L_i = - (y_i \ln f(x_i) + (1-y_i) \ln (1-f(x_i)))
$$

- **规避梯度弥散**

    可以证明，$$\displaystyle \frac{\partial L_i}{\partial x_i} = A \cdot (f(x_i)-y_i)$$，即梯度受误差 $$(f(x_i)-y_i)$$ 控制。
    
    因此，交叉熵损失函数具有 “误差越大，梯度下降越快” 的良好性质，可以完美解决MSE梯度下降过慢的问题。
    
    > 1. 之前提到，分类模型会存在一个转换函数 $$\sigma(x)$$ 将数值转化到 [0,1] 的概率区间。典型的，$$\sigma = sigmoid$$，可以证明 $$\sigma^{'}(x) = \sigma(x)(1-\sigma(x))$$；
    > 2. 令 $$f(x) = \sigma(g(x))$$，其中 $$g(x)$$ 为拟合函数；
    > 3. $$\displaystyle \frac{\partial L_i}{\partial x_i} = \frac{\partial L_i}{\partial f(x_i)} \frac{\partial L_i}{\partial f(x_i)} \frac{\partial g(x_i)}{\partial x_i} =...= (f(x_i)-y_i) g^{'}(x)$$；完整证明可参考 [交叉熵损失函数](https://zhuanlan.zhihu.com/p/35709485)

- **KL散度**

    交叉熵的公式是基于 KL散度 推导的。KL散度，也叫相对熵，是用于衡量同一随机变量 $$x$$ 两个分布 $$p(x)$$ 和 $$q(x)$$ 之间的差异。假设以 $$p(x)$$ 作为真实分布，$$q(x)$$ 作为预测分布，有
    
    $$
    D(p|q) = \sum_{i=1}^n p(x_i) \ln \frac{p(x_i)}{q(x_i)}
    $$
    
    其中 $$n$$ 表示可能取值的数量，例如二分类时 $$n=2$$。 将上述公式中对数除法拆开，即为交叉熵公式。完整证明可以参考 [交叉熵](https://zhuanlan.zhihu.com/p/61944055)。

- **概率解释**

    交叉熵 同样存在概率解释。关键是理解概率公式：
    
    $$
    P(y_i|x_i) = f(x_i)^{y_i}(1-f(x_i))^{1-y_i}
    $$

    转化为对数似然函数即为交叉熵。

### 1.3. Log

$$
\displaystyle L_i = - \ln \Big(y_if(x_i) + (1-y_i)(1-f(x_i)\Big) $$ 

对数损失函数常用于逻辑回归问题中。

## 2. Regression

回归模型，一般使用 MSE 作为损失函数。

