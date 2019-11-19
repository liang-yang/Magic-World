<!-- toc -->

# Neural Network

## 1. Structure

神经网络主要由 输入层，隐藏层，输出层 构成，每一层都包含多个神经元。

<div align=center>![](https://tva1.sinaimg.cn/large/006y8mN6gy1g8whem5p59j30d00bsdg0.jpg)

- **输入层**：输入层不做数据变换，仅将每个输入特征转化为神经元结构；    
- **隐藏层**：神经网络主要的数据变换都在隐藏层。隐藏层可以包含多层，每层都包含多个神经元；    
- **输出层**：隐藏层输出的数据在输出层真正转化为业务功能。例如，如果神经网络用于分类模型，那么在输出层需要将隐藏层的输出转化为分类概率（转化函数，二分类一般使用sigmoid，多分类一般使用softmax）。如果神经网络用于回归，那么在输出层需要将隐藏层的输出转化为拟合的回归值；    

## 2. Neuron

神经元基于输入神经元的线性组合得到 z 值，再通过 activation function 将 z 值转化为 a 值，a 值即为此神经元的输出。

<div align=center>![](https://tva1.sinaimg.cn/large/006y8mN6gy1g8wik1wlkcj30ac07vdfp.jpg)

## 3. Feed Forward

符号 | 含义
:-: | :-: 
$$w^l$$ | $$l$$ 层权重矩阵
$$b^l$$ | $$l$$ 层偏置向量
$$z^l$$ | $$l$$ 层带权输入
$$\sigma$$ | 激活函数
$$a^l$$ | $$l$$ 层激活向量

$$
z^l = w^l \cdot a^{l-1} + b^l
$$

$$
a^l = \sigma(z^l) = \sigma(w^l \cdot a^{l-1} + b^l)
$$

## 4. Back Propagation

反向传播 是根据梯度下降算法，通过不断调整神经元的权重和偏置，最终得到代价函数取最小值时的权重和偏置。

因此，反向传播的核心是求**各神经元的权重和偏置的梯度**，主要通过如下 四 个公式求得。

符号 | 含义
:-: | :-: 
$$\odot$$ | Hadamard乘积，$$ \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} \odot \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} = \begin{bmatrix} a_1 * b_1 \\ a_2 * b_2 \end{bmatrix}$$
$$w^l_{jk}$$ | $$l$$ 层权重矩阵($$j,k$$)元素
$$b^l_j$$ | $$l$$ 层偏置向量$$j$$分量
$$z^l_j$$ | $$l$$ 层带权输入$$j$$分量
$$a^l_j$$ | $$l$$ 层激活向量$$j$$分量
$$\delta^l = \frac{\partial C}{\partial z^l} $$ | $$l$$ 层误差向量
$$\delta^l_j$$ | $$l$$ 层误差向量$$j$$分量
$$L$$ | 输出层
$$C$$ | 代价函数

- **公式一**：$$\displaystyle \delta^L = \frac{\partial C}{\partial a^L} \odot \sigma^{'}(z^L)$$

   > $$\displaystyle \delta_j^l = \frac{\partial C}{\partial z_j^l} = \frac{\partial C}{\partial a_j^l} \frac{\partial a_j^l}{\partial z_j^l} = \frac{\partial C}{\partial a_j^l} \sigma^{'}(z_j^l)$$

   > $$\displaystyle \delta^l = \frac{\partial C}{\partial z^l} = \frac{\partial C}{\partial a^l} \odot \sigma^{'}(z^l)$$

- **公式二**：$$\displaystyle \delta^l = ((w^{l+1})^T \cdot \delta^{l+1}) \odot \sigma^{'}(z^l)$$

   > $$\displaystyle z^{l+1}_k = \sum_j w^{l+1}_{kj}a^l_j+b^{l+1}_k = \sum_j w^{l+1}_{kj} \sigma (z^l_j)+b^{l+1}_k $$

   > $$\displaystyle \delta^l_j = \frac{\partial C}{\partial z^l_j} = \sum_k \frac{\partial C}{\partial z^{l+1}_k} \frac{\partial z^{l+1}_k}{\partial z^l_j} = \sum_k \delta^{l+1}_k \frac{\partial z^{l+1}_k}{\partial z^l_j} = \sum_k \delta^{l+1}_k w^{l+1}_{kj} \sigma' (z^l_j) $$

- **公式三**：$$\displaystyle \frac{\partial C}{\partial b^l} = \delta^l$$

   > $$\displaystyle \frac{\partial C}{\partial b_j^l} = \frac{\partial C}{\partial z_j^l} \frac{\partial z_j^l}{\partial b_j^l} = \frac{\partial C}{\partial z_j^l} * 1 = \frac{\partial C}{\partial z_j^l} = \delta^l_j$$

- **公式四**：$$\displaystyle \frac{\partial C}{\partial w_{jk}^l} = \delta^l_j a_k^{l-1}$$

   > $$\displaystyle \frac{\partial C}{\partial w_{jk}^l} = \frac{\partial C}{\partial z_j^l} \frac{\partial z_j^l}{\partial w_{jk}^l} = \frac{\partial C}{\partial z_j^l} \frac{\partial (w_{jk}^l a_k^{l-1}+b_j^l)}{\partial w_{jk}^l} = \delta^l_j a_k^{l-1}$$

## 5. Activation Function

### 5.1. Use For

“激活”这个名字来自于生物上的神经网络结构。在每个神经元中，需要一个“开关”来决定该神经元的信息是否会被传递到其相连的神经元去，这个“开关”在这里也就是激活函数。

在神经网络中，激活函数还提供了非线性能力。否则，无论多少层的网络结构都可以用一个单层线性网络来代替。因此，激活函数又称作非线性映射函数。

输出层，激活函数将神经网络的输出映射到最终预测结果。

### 5.2. Function List

理论上，所有的非线性函数都可以作为激活函数（后面我们会看到，还有单调递增的要求），常见的有：Sigmoid, Tanh, ReLU 等。

<div align=center>![](https://tva1.sinaimg.cn/bmiddle/006y8mN6gy1g8yvopznvoj30hs0dcq37.jpg)

激活函数 | 公式 | 是否平滑 | 是否原点中心 | 是否梯度弥散 
:-: | :-: | :-: | :-: | :-: 
Step | $$\displaystyle \begin{cases} 0 &\text{if } x \leq 0 \\ 1 &\text{if } x > 0 \end{cases} $$ | 否 | 否 | - 
Sigmoid | $$\displaystyle \frac{1}{1+e^{-x}} $$ | 是 | 否 | 是  
Tanh | $$\displaystyle \frac{e^x-e^{-x}}{e^x+e^{-x}}$$ | 是 | 是 | 是 
ReLU | $$\displaystyle \max(0, x) $$ | 是 | 否 | 否 

> 1. 函数非线性，是指其导数不恒等，否则神经网络不具备非线性拟合能力；
> 2. 函数平滑，是指其导数连续且不恒为零，否则无法梯度下降；

### 5.3. Zero Center

> Reference    
> -- [Zero Center Activation Function](https://liam.page/2018/04/17/zero-centered-active-function/)

原点中心，是指激活函数的值域包含正、负数。例如sigmoid的值域为(0,1)，仅可取正数，就不是原点中心的。 

**原点中心的好处，是在梯度下降时收敛速度更快。** 

> 1. 根据【Back Propagation】公式四：$$\displaystyle \frac{\partial C}{\partial w_{jk}^l} = \delta^l_j a_k^{l-1}$$，神经元输入权重的梯度正负号由 $$\delta^l_j$$ 和 $$a^{l-1}_k$$ 决定；
> 2. 对于单个神经元来说，$$\delta^l_j$$ 是固定的；
> 3. 假设激活函数非原点中心，例如 sigmoid 其输出恒大于0，即 $$a^{l-1}_k$$ 恒大于0；
> 4. 因此，对于单个神经元来说，其所有输入权重的梯度的正负号相同，即下降方向相同。此时，假设各输入权重的最优解并不是相同的正负号，那么梯度下降的过程就会比较曲折，收敛速度比较慢，如下图所示。

<div align=center>![](https://tva1.sinaimg.cn/bmiddle/006y8mN6gy1g8ypkpfhk3j30i80gc3yp.jpg)

### 5.4. Gradient Vanishing

> Reference    
> -- [Gradient Vanishing](https://www.cnblogs.com/yangmang/p/7477802.html)       
> -- [神经网络梯度消失和梯度爆炸及解决办法](https://mp.weixin.qq.com/s/6xHC5woJND14bozsBNaaXQ) 

梯度弥散，也叫梯度消失，是指靠近输入层的神经元的梯度非常小，几乎接近于0，导致参数几乎无法学习。

**造成梯度弥散的根本原因，是由于 激活函数 的 饱和性。 **根据【Back Propagation】公式二：$$\displaystyle \delta^l = ((w^{l+1})^T \cdot \delta^{l+1}) \odot \sigma^{'}(z^l)$$，对其递归转换，可知 $$\delta^l$$ 受 $$\sigma^{'}$$ 的指数级影响。那么当 $$\sigma^{'} < 1$$ 时，尤其当 $$\sigma^{'} \to 0$$ 时，$$\delta^l$$ 会逐渐趋近于 0，即梯度弥散。

> 1. 函数的饱和性，是指函数的导数趋近于0。需要注意，这里导数为0是指的激活函数，而不是代价函数，所以导数为0与最优解没有直接关系；
> 2. 实际上，根据上面的公式，是否梯度弥散还与权重 $$w$$ 有关，因为 $$\delta$$ 实际是受 $$w$$ 和 $$\sigma^{'}$$ 的乘积的指数级影响。但 $$w$$ 一般初始取值较小，变化幅度也较小，更多还是考察 $$\sigma^{'}$$；
> 3. 以 sigmoid 的为例，其导数值域为 (0, 1/4)，曲线如下图所示。当z值很大或很小时，$$\sigma^{'}$$ 趋近于0，导致梯度弥散；
> 4. 为规避激活函数饱和性的问题，一方面可以更换激活函数，一方面可以通过 normalization 减小每层的 z 向量，使得其偏离饱和区间；

<div align=center>![](https://tva1.sinaimg.cn/bmiddle/006y8mN6gy1g92aca8mmkj30hs0dcgls.jpg)

与 梯度弥散 相对的，如果 $$\sigma^{'} > 1$$，$$\delta^l$$ 会指数级增长，形成 梯度爆炸（Gradient Exploding）。

因此，从规避 梯度弥散 和 梯度爆炸 的角度，活跃函数的导数最好为 1，但导数恒为 1 又不具备了非线性。由此，引入了 ReLU及其系列 的激活函数。

### 5.5. ReLU

ReLU，Rectified Linear Unit，修正线性单元，虽然简单，却是目前最为常用的 激活函数：

- 在正区间规避了 梯度弥散 的问题；
- 计算速度非常快，不需要计算指数；
- 在正区间导数恒为1，大于 Sigmoid/Tanh，因此收敛速度更快；

当然，ReLU 也存在如下问题：

- 非 Zero Center；
- Dead ReLU Problem，由于在负区间函数值恒为0，使得一旦某神经元的 z 值为负数时，就无法对最终输出提供信息，即神经元进入Dead状态。且由于此时其导数也恒为0，即 z 值无法再更新，神经元会永远处于Dead状态；

为了规避 ReLU 的问题，提出了 Leaky ReLU，ELU(Exponential Linear Unit) 等优化方案（虽然理论上优于ReLU，但在实际使用中目前并没有证明它们总是优于ReLU）。

激活函数 | 公式  
:-: | :-:  
ReLU | $$\displaystyle \max(0, x) $$  
Leaky ReLU | $$\displaystyle \max(0.01x, x) $$  
ELU | $$\displaystyle \begin{cases} \alpha (e^x-1) &\text{if } x \leq 0 \\ x &\text{if } x > 0 \end{cases} $$ 

<div align=center>![](https://tva1.sinaimg.cn/bmiddle/006y8mN6gy1g93dtuoliij30hs0dc0sw.jpg)

### 5.6. Maxout

> Reference    
> -- [GoogLeNet, Maxout and NIN](https://zhuanlan.zhihu.com/p/42704781)     

Maxout 的数学表达式如下：

$$
Maxout(k, x) = \max(w_1^Tx+b_1, w_2^Tx+b_2, \cdots, w_k^Tx+b_k)
$$

资料中常见的函数图像如下：

<div align=center>![](https://tva1.sinaimg.cn/large/006y8mN6gy1g93e8no44kj30k004wwel.jpg)

> 1. 图1，$$k=2$$，2条直线，$$y=0,\ y=x$$，这也就是ReLU函数；
> 2. 图2，$$k=2$$，2条直线，$$y=-x,\ y=x$$；
> 3. 图3，$$k=5$$，5条直线，从左到右分别取了绿色红色青色紫色黄色的各一段，都是对应区间里五条线性函数的最大值；

当 $$k$$ 足够大时，Maxout 可以以任意小的精度逼近任何凸函数。

Maxout 的问题是，网络的参数是其他激活函数的 $$k$$ 倍，但没有带来等价的精度提升。

### *5.7. Softmax

$$
softmax(z_k) = \frac{e^{z_k}}{\sum_{i=1}^n e^{z_i}}
$$

<div align=center>![](https://tva1.sinaimg.cn/large/006y8mN6gy1g93fnryvlij30iu0az74t.jpg)

它只会被用在网络中的最后一层，用来进行最后的分类和归一化。


https://www.zhihu.com/search?type=content&q=Softmax

https://www.zhihu.com/question/23765351/answer/240869755

https://zhuanlan.zhihu.com/p/25723112



## *6. Cost Function

### *6.1. Quadratic

二次代价函数

$$
C = \frac{1}{2n} \sum_{i=1}^n (y_i-a^L(x_i))^2
$$

### *6.2. Cross Entropy

交叉熵代价函数

$$
C = - \frac{1}{n} \sum_{i=1}^n (y \ln a^L(x_i) + (1-y) \ln (1-a^L(x_i)))
$$









## 7. Why use Nerual Network

> Reference    
> -- [神经网络的理解与实现](https://www.cnblogs.com/lliuye/p/9183914.html)    
> -- [理解神经网络的激活函数](https://zhuanlan.zhihu.com/p/36763712)    

万能逼近（universal approximation）定理：








---




    
- 梯度下降的动态学习步长；

- 梯度下降的限制条件；

- 怎么样初始化参数？






Dropout

neurons saturated



一般现在的权重初始化是用Xavier。


<div align=center>![](https://tva1.sinaimg.cn/large/006y8mN6gy1g8ym3oc7gxj30yb0u042c.jpg)

## Reference





