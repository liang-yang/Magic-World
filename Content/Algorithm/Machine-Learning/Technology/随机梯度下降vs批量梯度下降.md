<!-- toc -->

# 随机梯度下降 vs 批量梯度下降

---

> see [梯度下降](https://zhuanlan.zhihu.com/p/36902908)

## 1. 最优化问题

最优化问题是求解函数极值的问题，包括极大值和极小值。

> 为统一分析，我们一般把求极大值转化为求极小值，如 $$ \max \big( f(x) \big) = \min \big( -f(x) \big).$$

一般来说，我们是通过微积分，对函数求导数为0的点。因为 导数为0 是 极值点 的必要条件（非充分条件）。

> 需要注意，一个函数可能有多个局部极值点，我们需要通过比较这些局部极值点才能找到全局极值点。

## 2. 梯度下降(Gradient Descent)

但是，当函数特别复杂时，求导为零很困难。例如：

$$ 
\qquad f(x, y) = x^3 - 2x^2 + e^{xy} - y^3 + 10y^2 + 100 \sin (xy) 
$$

这个函数的求导为零就很难求解。由此，引入了 梯度下降 方法。

## 2.1. 梯度

**梯度** 是 多元函数 对各个自变量偏导数形成的向量，定义为：

$$
\qquad \nabla f(x_1,x_2,...,x_k) = \bigg( \frac{\partial f}{\partial x_1},\frac{\partial f}{\partial x_2},...,\frac{\partial f}{\partial x_k} \bigg)^T
$$

其中 $$ \nabla $$ 称为梯度算子，它作用于一个多元函数，得到一个向量。 例如 $$ \nabla(x^2+xy-y^2) = (2x+y,x-2y) $$。

可导函数在某一点处取得极值的必要条件是梯度为零，梯度为零的点称为函数的驻点，这是疑似极值点。   
需要注意，梯度为零只是函数取极值的必要条件而不是充分条件，即梯度为零的点可能不是极值点。如果需要确定是否极值点，还需要观察二阶导数：
- 如果二阶导数大于0，函数有极小值；
- 如果二阶导数小于0，函数有极大值；
- 如果二阶导数等于0，情况不定。

## 2.2. 梯度下降

对于复杂的函数，我们可以通过 迭代法，从一个初始点 $$ x_0 $$，反复使用某种规则从 $$ x_{k} $$ 移动到下一个点 $$ x_{k+1} $$，即

$$
\qquad x_{k+1} = h(x_{k})
$$

那么，具体的迭代规则是什么呢？

我们将多元函数 $$ f(x) $$ 在 $$ x_0 $$ 点处泰勒展开，有：

$$
\qquad f(x_0 + \Delta x) = f(x_0) + \big( \nabla f(x_0) \big)^T \Delta x + \omicron ( \Delta x )
$$

当 $$ \Delta x $$ 足够小时，我们可以忽略高次项 $$ \omicron ( \Delta x ) $$，可以得到：

$$
\qquad f(x_0 + \Delta x) - f(x_0) \approx \big( \nabla f(x) \big)^T \Delta x
$$

假设我们现在要求 极小值（求极大值可以转化为求极小值），则需要令函数递减：

$$
\qquad f(x_0 + \Delta x) - f(x_0) \approx \big( \nabla f(x) \big)^T \Delta x \leqslant 0
$$

即

$$
\qquad \big( \nabla f(x) \big)^T \Delta x = || \big( \nabla f(x) \big)^T || \cdot || \Delta x || \cdot \cos \theta \leqslant 0
$$

其中，$$ \theta $$ 是向量 $$ \big( \nabla f(x) \big)^T $$ 和 $$ \Delta x $$ 的夹角。 那么，只要 $$ \cos \theta \leqslant 0 $$ 即可。 

特别的，在 $$ \theta = \pi $$，即向量 $$ \Delta x $$ 与梯度 $$ \big( \nabla f(x) \big)^T $$ 反向时，$$ \cos \theta = -1 $$，就可以更快的朝极小值移动。因此，我们令

$$ 
\qquad \Delta x = - \eta \cdot \big( \nabla f(x) \big)^T 
$$

则必有 $$ \cos \theta = -1 $$。其中，$$ \eta $$ 我们称之为 步长，以此控制 $$ \Delta x $$ 足够小。








$$ \eta $$


那么，迭代到什么时候停止呢？ 理想情况下，迭代到 梯度 为零的点即为极值点。







假设拟合函数为 $$ h(\theta_1, \theta_2, ... , \theta_n)$$
$$


$$






