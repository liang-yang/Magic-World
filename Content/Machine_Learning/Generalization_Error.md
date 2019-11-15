<!-- toc -->

# Generalization Error

---

## 1. Generalization Abilitiy

在分析 Generalization Error（泛化误差）之前，先要理解Generalization Abilitiy（泛化能力）。 首先，我们定义两个概念：

- **Model Space**：类似于线性代数里的向量空间，Model Space 是指某一类 Model 的集合。例如线性模型 $$y = w \cdot x + b$$ 其实代表了平面上所有的直线，因此严格的说，$$y = w \cdot x + b$$ 是 线性模型空间。同理，$$y = a \cdot x^2 + b \cdot x + c$$ 是 二次曲线模型空间。
> 以上讨论没有考虑 超参数（Hyper-Parameters）。实践中，不同的超参数对应不同的 Model Space。

- **Model**：Model 就是 Model Space 中的一个元素，如线性模型空间中的某一条直线，就是一个 Model。

在机器学习中，一般我们会选择一个 **Model Space**，再定义 **Loss Function**，通过在样本集上最小化 **Loss Function** 从 **Model Space** 中选择一个元素作为结果 **Model**，这个过程就是 **Model Fit**（模型拟合）。 那么，如果我们的样本集足够大，甚至直接是总体，那只要我们选择了合适的 **Model Space** 和 **Loss Function**，通过 **Model Fit** 得到的结果 **Model** 在样本集上一般会有很好的表现。

但是，现实中一般只能获取部分样本数据。因此，**Model Fit** 只能保证结果 **Model** 在这部分样本数据上表现良好，在未知数据上的表现是不确定的。而实际上，我们更看重 **Model** 在未知数据上的表现。**Model** 的这种在未知数据上的表现我们就称作泛化能力（Generalization Abilitiy）。   

机器学习中，存在一个 **Model Selection** 的步骤，其实就是选择泛化能力最强的模型。

## 2. Generalization Error

严格来讲，由于我们无法获取样本的总体，很难绝对准确的评估模型的泛化能力。因此，在实际项目中，我们一般通过随机划分 训练集 和 测试集，以训练集所拟合的模型在测试集上的预测值与真实值的误差，从统计意义上评估模型的泛化能力。这个误差，我们就称作 泛化误差（Generalization Error）。

那么，如果某训练集拟合出的模型在某测试样本上的泛化误差小，是不是就认为此模型的泛化能力强呢？答案是否定的。因为测试集上的泛化误差只是从 统计意义 上评估模型的泛化能力，因此 泛化误差 也需要通过统计意义来表示。  
抽象来看，一次拟合得到的泛化误差其实也是一个样本。根据统计学理论，基于样本估计总体，一般会分解为 均值、方差 等特征值来分析。模型的泛化误差也会进行类似的分析，这就是 Bias-Variance Decomposition（偏差-方差分解）。

## 3. Bias-Variance Decomposition

### 3.1. Bias and Variance

在介绍概念之前，我们先图形化的感受下 Bias 和 Variance。 我们以 靶心图 来模拟通过多个不同的训练样本集拟合模型，并预测同一测试样本的场景。 靶心作为测试样本的真实值，每一次的预测类似于一次打靶。下图表示 高、低偏差 和 高、低方差 的四种不同组合。

![](https://tva1.sinaimg.cn/large/006y8mN6gy1g72ld496kjj30cw0cht96.jpg)

下面给出 Bias 和 Variance 的概念：

- Bias，偏差，测试样本的预测值 与 测试样本的真实值 的误差的均值（数学期望）；
- Variance，方差，测试样本的多个预测值的方差（与 测试样本的真实值 无关）；

直观感觉，Bias 表示预测是否准确，Variance 表示预测是否稳定。

### 3.2. Mathematical 

下面我们给出数学公式，部分符号如下表所示。

符号 | 涵义 
:-: | :-: 
$$x$$ | 测试样本 
$$Y$$ | $$x$$ 的真实值 
$$f$$ | 训练得到的模型 
$$f(x)$$ | 模型对 $$x$$ 的预测值 
$$E[t]$$ | $$t$$ 的数学期望 

泛化误差 $$Error$$、偏差 $$Bias$$、方差 $$Variance$$ 的数学公式分别为：

$$
Error = E[(f(x) - Y)^2]
$$

$$
Bias = E[f(x)] - Y
$$

$$
Variance = E[(f(x) - E[f(x)])^2]
$$

可以证明：

$$
Error = Bias^2 + Variance
$$

> 证明过程中，关键点为 $$E[f(x)-E[f(x)]] = 0$$。

个人认为，由于公式中指代的多为数学期望，因此在应用中无实际意义。但在理论分析中，还是有意义的。例如，通过公式我们可以看出，通过降低 Bias 和 Variance，都可以降低泛化误差。

需要注意，在以上的讨论中未考虑样本的观测值与真实值之间的差异，即 噪声$$\varepsilon$$ 的影响。 

### 3.3. Analytical Bias and Variance

我们尝试分析 Bias 和 Variance 的本质，即为什么会产生 Bias 和 Variance：

- Bias，是模型预测值与真实值的误差。那么，为什么会产生误差呢？是由于模型无法完美的拟合数据。一般存在三方面的原因：
    1. **噪声影响**。训练集的样本噪声会导致模型拟合不准确。即使模型拟合准确，测试集的样本噪声又会使预测结果感觉不准确；
    2. **训练集样本数量不足**。当训练样本不足时，特征不够突出，以统计学为基础的机器学习模型就无法很好的拟合数据。这种情况下需要增加样本数据；
    3. **模型选择的不合适**，即前文的 Model Sapce 选择不合适。例如曲线分布的数据选择线性模型去拟合，那怎么都是不合适的。这种情况下需要改变所选择的 Model Sapce；
- Variance，是模型多次预测值的方差（需要注意，这里的多次预测使用的是相同的 Model Space，仅训练样本发生了改变）。存在方差说明多次预测结果产生了变化，那么，为什么会产生变化呢？本质来说是由于 **噪声影响**。每一次训练样本不一样，噪声也会不一样，拟合的结果模型也不一样，预测结果自然也就不一样。因此，小的 Variance 是不可避免的。我们这里重点分析导致 Variance 较大的三个原因：
    1. **训练集中异常样本较多**。异常样本，是指样本观测值与真实值差异较大的样本，即噪声较大的样本。这样，噪声越大模型拟合差异越大，Variance 自然越大；
    2. **训练集样本数量不足**。训练集样本越少，单个训练样本的噪声对模型拟合的影响越大，Variance 也就越大；
    3. **模型选择的过于复杂**。理论上，模型越复杂，学习能力越强，对训练集的拟合效果越好。但是，学习能力过强的模型会把噪声也完整的学习进去，导致 Variance 较大；

基于上述分析，我们有以下两个步骤可以同时降低 Bias 和 Variance：

1. **增加训练集样本数量**。训练集样本数量越多，特征越明显，单样本的噪声影响越小。
> 但是，训练集样本数量越多，计算量越大。因此，还是要适当平衡 泛化误差 和 训练集样本数量。
2. **剔除训练集中的异常样本**。

### 3.4. UnderFit vs OverFit

根据上面的分析，Bias 和 Variance 的产生原因都存在 模型选择不合适。但是，二者所指的 不合适 的意义并不一样，准确的说是互相矛盾的。理论上，模型越复杂，学习能力越强，模型拟合越准确。但同时噪声也会被学习的越多，所以 Variance 越大。也就是说，随着模型复杂度的增加，Bias 减小，Variance 增大，如下图所示。

![](https://ww2.sinaimg.cn/large/006y8mN6gy1g6lkh7580qj30do08lmx8.jpg)

需要注意，这里的 Bias 为数学期望，不能单纯的追求低 Bias，这在实际建模中并无意义。 因此，我们希望找到合适的模型复杂度，均衡 Bias 和 Variance，得到最小的泛化误差。

同时，我们也可以得到如下结论：**Bias 源于欠拟合，Variance 源于过拟合。**

## 4. Bagging and Boosting

首先，给出结论：**Bagging 可降低 Variance，Boosting 可降低 Bias**。

### 4.1. Bagging

基于前面的分析，复杂模型的学习能力很强，容易过拟合，所以存在Bias的数学期望较小但Variance较大的问题。那么，我们有没有办法降低Variance使得每次的Bias更接近数学期望呢？ 在统计学中，我们为了获取更精准的特征值，经常会求多次特征值再取均值的方式减小误差，这里我们同样可以采用这种思路，这就是 Bagging。 Bagging 通过对训练样本有放回的循环采样，对每次循环采样的子样本集训练一个子模型，以所有子模型的预测值的均值作为 Bagging 的最终预测值。 由于子样本集的相似性，各子模型应具有相同的 Bias 和 Variance。假设每个子模型的预测值为 $$Y_i$$，那么 Bagging 后的 Bias 和 Variance 分别为：

$$
Bias = E[\frac{\sum Y_i}{n}] = E[Y_i]
$$

$$
Variance = Var(\frac{\sum Y_i}{n}) = \frac{Var(\sum Y_i)}{n^2} = \frac{n \cdot Var(Y_i)}{n^2} = \frac{Var(Y_i)}{n}
$$

> 1. 以上 $$Variance$$ 公式成立的前提条件是各个子模型相互独立的极端情况。概率论中，相互独立的两个随机变量，协方差为零，和的方差等于方差之和，即 $$Var(A+B) = Var(A) + Var(B)$$。
> 2. 再考虑各个子模型完全不独立，即所有子模型完全一致的极端情况。此时使用的公式为 $$Var(k \cdot A) = k^2 \cdot Var(A)$$。此种场景下，Bagging 无法降低方差。
> 3. 一般来说，实际场景会处于以上两个极端情况之间，因此 Bagging 降低方差的效果也在二者之间。
    
通过以上公式可以看出，通过 Bagging 对 Bias 无影响，但可以降低 Variance，使得每次预测会更接近于 Bias，进而得到更小的泛化误差。

### 4.2. Boosting

关于 “Boosting 可降低 Bias”，完整的逻辑分析和数学证明还存疑，暂遗留。

简单的说，Boosting 是迭代算法，每一次迭代都根据上一次迭代的预测结果对样本进行权重调整，所以随着迭代不断进行，误差会越来越小，所以模型的 Bias 会不断降低。

## Reference

- [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html)
- [Bias and Variance](https://zhuanlan.zhihu.com/p/38853908)
- [知乎：为什么说bagging是减少variance，而boosting是减少bias?](https://www.zhihu.com/question/26760839)
- [知乎：机器学习中的Bias(偏差)，Error(误差)，和Variance(方差)有什么区别和联系？](https://www.zhihu.com/question/27068705)
