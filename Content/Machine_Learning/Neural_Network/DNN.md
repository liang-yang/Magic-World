<!-- toc -->

# DNN

DNN，Deep Neural Network，深度神经网络，最基础的神经网络。

## 1. Neuron and Neural Network

### 1.1. Neural

神经元 是神经网络最基本的组成元素。

神经元 基于 输入神经元 的线性组合得到 z 值，再通过 activation function 将 z 值转化为 a 值，a 值即为此神经元的输出。

<div align=center>![](https://tva1.sinaimg.cn/large/006y8mN6gy1g8wik1wlkcj30ac07vdfp.jpg)

### 1.2. Neural Network

神经网络主要由 输入层，隐藏层，输出层 构成，每一层都包含多个神经元。

<div align=center>![](https://tva1.sinaimg.cn/large/006y8mN6gy1g8whem5p59j30d00bsdg0.jpg)

> **输入层**：输入层不做数据变换，仅将每个输入特征转化为神经元结构；    
> **隐藏层**：神经网络主要的数据变换都在隐藏层。隐藏层可以包含多层，每层都包含多个神经元；    
> **输出层**：隐藏层输出的数据在输出层真正转化为业务功能。例如，如果神经网络用于分类模型，那么在输出层需要将隐藏层的输出转化为分类概率；如果神经网络用于回归模型，那么在输出层需要将隐藏层的输出转化为拟合的回归值；    

## 2. Feed Forward

前馈，是指从 输入层 开始，以前一层的输出作为后一层的输入来计算后一层的输出，层层传递，直到最后的 输出层。

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

## 3. Back Propagation

神经网络使用 梯度下降算法 训练参数。 由于网络层的架构，神经网络 需要通过代价函数计算输出层的参数梯度，再通过输出层的参数梯度计算倒数第二层的参数梯度，层层反向传递，最终得到第一层的参数梯度。 因此，这个学习过程叫做 反向传播。

反向传播 的核心是求**各神经元的权重和偏置的梯度**，主要通过如下 四 个公式求得。

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

## 4. Activation Function

“激活”这个名字来自于生物上的神经网络结构。在每个神经元中，需要一个“开关”来决定该神经元的信息是否会被传递到其相连的神经元去，这个“开关”在这里也就是激活函数。

在神经网络中，激活函数还提供了非线性能力。否则，无论多少层的网络结构都可以用一个单层线性网络来代替。因此，激活函数又称作非线性映射函数。

> 函数非线性，是指其导数不恒为某值。

输出层，激活函数将神经网络的输出映射到最终预测结果。

### 4.1. Function List

常见的激活函数有：Sigmoid, Tanh, ReLU 等。

<div align=center>![](https://tva1.sinaimg.cn/bmiddle/006y8mN6gy1g8yvopznvoj30hs0dcq37.jpg)

激活函数 | 公式 | 是否平滑 | 是否原点中心 | 是否梯度弥散 
:-: | :-: | :-: | :-: | :-: 
Step | $$\displaystyle \begin{cases} 0 &\text{if } x \leq 0 \\ 1 &\text{if } x > 0 \end{cases} $$ | 否 | 否 | - 
Sigmoid | $$\displaystyle \frac{1}{1+e^{-x}} $$ | 是 | 否 | 是  
Tanh | $$\displaystyle \frac{e^x-e^{-x}}{e^x+e^{-x}}$$ | 是 | 是 | 是 
ReLU | $$\displaystyle \max(0, x) $$ | 是 | 否 | 否 

> 函数平滑，是指其导数连续且不恒为零，否则无法梯度下降；

### 4.2. Zero Center

> Reference    
> -- [Zero Center Activation Function](https://liam.page/2018/04/17/zero-centered-active-function/)

原点中心，是指激活函数的值域包含正、负数。例如sigmoid的值域为(0,1)，仅可取正数，就不是原点中心的。 

**原点中心的好处，是在梯度下降时收敛速度更快。** 

> 1. 根据【Back Propagation】公式四：$$\displaystyle \frac{\partial C}{\partial w_{jk}^l} = \delta^l_j a_k^{l-1}$$，神经元输入权重的梯度正负号由 $$\delta^l_j$$ 和 $$a^{l-1}_k$$ 决定；
> 2. 对于单个神经元来说，$$\delta^l_j$$ 是固定的；
> 3. 假设激活函数非原点中心，例如 sigmoid 其输出恒大于0，即 $$a^{l-1}_k$$ 恒大于0；
> 4. 因此，对于单个神经元来说，其所有输入权重的梯度的正负号相同，即下降方向相同。此时，假设各输入权重的最优解并不是相同的正负号，那么梯度下降的过程就会比较曲折，收敛速度比较慢，如下图所示。

<div align=center>![](https://tva1.sinaimg.cn/bmiddle/006y8mN6gy1g8ypkpfhk3j30i80gc3yp.jpg)

### 4.3. Gradient Vanishing

> Reference    
> -- [Gradient Vanishing](https://www.cnblogs.com/yangmang/p/7477802.html)       
> -- [神经网络梯度消失和梯度爆炸及解决办法](https://mp.weixin.qq.com/s/6xHC5woJND14bozsBNaaXQ) 

梯度弥散，也叫梯度消失，是指靠近输入层的神经元的梯度非常小，几乎接近于0，导致参数几乎无法学习。

**造成梯度弥散的主要原因，是由于 激活函数 的 饱和性。 **根据【Back Propagation】公式二：$$\displaystyle \delta^l = ((w^{l+1})^T \cdot \delta^{l+1}) \odot \sigma^{'}(z^l)$$，对其递归转换，可知 $$\delta^l$$ 受 $$\sigma^{'}$$ 的指数级影响。那么当 $$\sigma^{'} < 1$$ 时，尤其当 $$\sigma^{'} \to 0$$ 时，$$\delta^l$$ 会逐渐趋近于 0，即梯度弥散。

> 1. 函数的饱和性，是指函数的导数趋近于0。需要注意，这里导数为0是指的激活函数，而不是代价函数，所以导数为0与最优解没有任何关系；
> 2. 实际上，根据上面的公式，是否梯度弥散还与权重 $$w$$ 有关，因为 $$\delta$$ 实际是受 $$w$$ 和 $$\sigma^{'}$$ 的乘积的指数级影响。但 $$w$$ 一般初始取值较小，变化幅度也较小，更多还是考察 $$\sigma^{'}$$；
> 3. 以 sigmoid 的为例，其导数值域为 (0, 1/4)，曲线如下图所示。当z值很大/小 或 网络层很深时，$$\sigma^{'}$$ 趋近于0，导致梯度弥散；
> 4. 为规避激活函数饱和性的问题，一方面可以更换激活函数，一方面可以通过 normalization 减小每层的 z 向量，使得其偏离饱和区间；

<div align=center>![](https://tva1.sinaimg.cn/bmiddle/006y8mN6gy1g92aca8mmkj30hs0dcgls.jpg)

与 梯度弥散 相对的，如果 $$\sigma^{'} > 1$$，$$\delta^l$$ 会指数级增长，形成 梯度爆炸（Gradient Exploding）。

因此，从规避 梯度弥散 和 梯度爆炸 的角度，活跃函数的导数最好为 1，但导数恒为 1 又不具备了非线性。由此，引入了 ReLU及其系列 的激活函数。

### 4.4. ReLU

ReLU，Rectified Linear Unit，修正线性单元，虽然简单，却是目前最为常用的 激活函数：

- 在正区间规避了 梯度弥散 的问题；
- 计算速度非常快，不需要计算指数；
- 在正区间导数恒为1，大于 Sigmoid/Tanh，因此收敛速度更快；

当然，ReLU 也存在如下问题：

- 非 Zero Center；
- Dead ReLU Problem，由于在负区间函数值恒为0，使得一旦某神经元的 z 值为负数时，就无法对最终输出提供信息，即神经元进入Dead状态。且由于此时其导数也恒为0，即 z 值无法再更新，神经元会永远处于Dead状态；

为了规避 ReLU 的问题，提出了 Leaky ReLU，ELU(Exponential Linear Unit) 等优化方案（虽然理论上优于ReLU，但在实际使用中目前并没有证明它们总是优于ReLU）。

另外，有研究指出 ReLU 更好的拟合了生物神经网络的稀疏性，使得其表现更好。具体可参考 [Deep Sparse Rectifier Neural Networks](https://www.cnblogs.com/neopenx/p/4453161.html)。

激活函数 | 公式  
:-: | :-:  
ReLU | $$\displaystyle \max(0, x) $$  
Leaky ReLU | $$\displaystyle \max(0.01x, x) $$  
ELU | $$\displaystyle \begin{cases} \alpha (e^x-1) &\text{if } x \leq 0 \\ x &\text{if } x > 0 \end{cases} $$ 

<div align=center>![](https://tva1.sinaimg.cn/bmiddle/006y8mN6gy1g93dtuoliij30hs0dc0sw.jpg)

### 4.5. Maxout

> Reference    
> -- [GoogLeNet, Maxout and NIN](https://zhuanlan.zhihu.com/p/42704781)     

Maxout 的数学表达式为：

$$
Maxout(k, x) = \max(w_1^Tx+b_1, w_2^Tx+b_2, \cdots, w_k^Tx+b_k)
$$

资料中常见的函数图像为：

<div align=center>![](https://tva1.sinaimg.cn/large/006y8mN6gy1g93e8no44kj30k004wwel.jpg)

> 1. 图1，$$k=2$$，2条直线，$$y=0,\ y=x$$，也就是ReLU函数；
> 2. 图2，$$k=2$$，2条直线，$$y=-x,\ y=x$$；
> 3. 图3，$$k=5$$，5条直线，从左到右分别取了绿红青紫黄的各一段，拟合了二次曲线函数；

当 $$k$$ 足够大时，Maxout 可以任意小的精度逼近任何凸函数。

Maxout 的问题是，网络的参数是其他激活函数的 $$k$$ 倍，但没有带来等价的精度提升，工程中不太使用。

### 4.6. Softmax

Softmax 一般用在网络中的最后一层，用来进行最后的分类和归一化。

$$
\displaystyle softmax(z_k) = \frac{e^{z_k}}{\sum_{i=1}^n e^{z_i}}
$$

<div align=center>![](https://tva1.sinaimg.cn/large/006y8mN6gy1g93fnryvlij30iu0az74t.jpg)

- **梯度**

   1. 假设 Softmax 的输入为向量 $$z=(z_1,z_2,...,z_k)$$，输出为向量 $$a = (a_1,a_2,...,a_k)$$；
   2. Softmax 的梯度可表示为：$$\displaystyle \frac{\partial a_j}{\partial z_i}$$，其中 $$i,j \in \{1,2,...,k\}$$。也就是说，完整梯度是一个矩阵；
   3. 当 $$i=j$$ 时，$$\displaystyle \frac{\partial a_j}{\partial z_i} = a_j(1-a_j)$$；
   4. 当 $$i \ne j$$ 时，$$\displaystyle \frac{\partial a_j}{\partial z_i} = -a_ja_i$$；
   5. 详细证明可参考 [Softmax函数与交叉熵](https://zhuanlan.zhihu.com/p/27223959)；

## 5. Why Nerual Network

理论证明，只要激活函数选择得当，神经元个数足够多，使用 3 层即包含一个隐含层的神经网络就可以实现对任何一个从输入向量到输出向量的连续映射函数的逼近。

- **几何解释**

   1. 隐藏层的每个神经元可以看作在输入空间中的超平面，例如输入层为两个变量，则隐藏层每个神经元可以视作二维平面上的一条直线；
   2. 通过 三 层神经网络，可以实现 与、或、非、亦或 等逻辑操作；
   3. 对 多条直线 进行 与、或、非、亦或 等逻辑操作，就可以实现对平面进行画块分类；
   4. 可参考 [理解神经网络的激活函数](https://zhuanlan.zhihu.com/p/36763712)；

   <div align=center>![](https://tva1.sinaimg.cn/large/006y8mN6gy1g9crqt9zrpj30dl09n74h.jpg)

- **理论证明**

   [万能逼近（universal approximation）定理](https://zhuanlan.zhihu.com/p/36763712)：如果 $$\varphi(x)$$ 是一个**非常数、有界、单调递增**的连续函数， $$I_{m}$$ 是 $$m$$ 维的单位立方体， $$I_{m}$$ 中的连续函数空间为 $$C(I_{m})$$。 对于任意 $$\varepsilon > 0$$ 以及 $$f\in C(I_{m})$$ 函数， 存在整数 $$N$$，实数 $$v_{i},b_{i}$$，实向量 $$w_{i}\in R^{m}$$，通过它们构造函数 $$F(x)$$ 作为函数 $$f(x)$$ 的逼近：
     
   $$
   F(x) = \sum^{N}_{i=1}v_i \varphi(w_i^Tx + b)
   $$
      
   对任意的 $$X\in R_{m}$$ 满足：    
   
   $$
   |F(x) - f(x)| < \varepsilon
   $$
   
   万能逼近定理的直观解释是可以构造出上面这种形式的函数，逼近定义在单位立方体空间中的任何一个连续函数到任意指定的精度。这个定理对激活函数的要求是必须非常数、有界、单调递增，并且连续。
