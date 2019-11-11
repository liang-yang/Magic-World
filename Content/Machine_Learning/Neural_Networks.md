<!-- toc -->

# Neural Network

神经网络(Neural Network)，是一种模仿生物神经网络（动物的中枢神经系统，特别是大脑）的结构和功能的数学模型或计算模型，用于对函数进行估计或近似。

## Structure

神经网络主要由 输入层，隐藏层，输出层 构成，每一层都包含多个神经元：

- **输入层**：输入层不做数据变换，仅将每个输入特征转化为神经元结构；
- **隐藏层**：神经网络主要的数据变换都在隐藏层。隐藏层可以包含多层，每层都包含多个神经元；
- **输出层**：隐藏层输出的数据在输出层真正转化为业务功能。例如，如果神经网络用于分类模型，那么在输出层需要将隐藏层的输出转化为分类概率（转化函数，二分类一般使用sigmoid，多分类一般使用softmax）。如果神经网络用于回归，那么在输出层需要将隐藏层的输出转化为拟合的回归值；

   ![](https://tva1.sinaimg.cn/large/006y8mN6gy1g8qimjxdywj30gk0b1aab.jpg)

## Neuron

神经元是神经网络中每一层的组成元素。每个神经元基于

## Activation Function

激活函数（activation function）是神经网络中最主要的组成部分之一。

“激活”这个名字来自于生物上的神经网络结构。 人工神经网络最初是受到生物上的启发而设计出了神经元这种单位结构，在每个神经元中，需要一个“开关”来决定该神经元的信息是否会被传递到其相连的神经元去，这个“开关”在这里也就是激活函数。

在神经网络中，激活函数还提供了非线性能力。否则，无论多少层的网络结构都可以用一个单层线性网络来代替。因此，激活函数又称作非线性映射函数。

在神经网络的输出层，激活函数还担任将前一层的输出映射到最终预测结果的任务。例如，对于一个二分类问题，通常最终输出层的激活函数就是sigmoid函数，而多分类任务则往往对应softmax函数。

---




- 偏置 b 有什么意义？怎么理解？

    从分类的角度，可以看作分类阈值；
    
- 梯度下降的动态学习步长；

- 梯度下降的限制条件；

- 怎么样初始化参数？

---

- Perceptron Neuron

$$
output = \begin{cases}
   0 &\text{if } \sum_j w_jx_j \leq threshold \\
   1 &\text{if } \sum_j w_jx_j > threshold
\end{cases}
$$

通过 $$ w \cdot x = \sum_j w_jx_j $$ 和 $$ b = - threshold $$ 的转换，可以得到：

$$
output = \begin{cases}
   0 &\text{if } w \cdot x + b \leq 0 \\
   1 &\text{if } w \cdot x + b > 0
\end{cases}
$$

- Sigmoid Neuron

Perceptron Neuron 中，$$output$$ 的变化并不平滑，这导致 权重$$w$$ 和 偏置$$b$$ 的微小变化不一定能体现到 $$output$$，不利于自动学习，因此引入 Sigmoid Neuron：

$$
output = Sigmoid(z) = \frac{1}{1+e^{-z}} = \frac{1}{1+e^{-(w \cdot x + b)}}
$$

Sigmoid 相对 Perceptron 就平滑的多，这样 权重$$w$$ 和 偏置$$b$$ 的微小变化会体现到 $$output$$ 的变化，更利于自动学习。

![](https://tva1.sinaimg.cn/large/006y8mN6gy1g8il3ob1f1j30hs0dc0sv.jpg)

梯度下降算法

随机梯度下降算法



权重矩阵 $$\displaystyle w^l$$

偏置向量 $$\displaystyle b^l$$

带权输入 $$\displaystyle z^l = w^l \cdot a^{l-1} + b^l$$

激活函数 $$\displaystyle \sigma$$

激活向量 $$\displaystyle a^l = \sigma(z^l) = \sigma(w^l \cdot a^{l-1} + b^l)$$



Hadamard 乘积(Schur 乘积)

$$
\begin{bmatrix} 1 \\ 2 \end{bmatrix}
\odot
\begin{bmatrix} 3 \\ 4 \end{bmatrix}
= 
\begin{bmatrix} 1 * 3 \\ 2 * 4 \end{bmatrix}
=
\begin{bmatrix} 3 \\ 8 \end{bmatrix}
$$

$$
\delta_j^l = \frac{\partial C}{\partial z_j^l} = \frac{\partial C}{\partial a_j^l} \cdot \frac{\partial a_j^l}{\partial z_j^l} = \frac{\partial C}{\partial a_j^l} \cdot \sigma^{'}(z_j^l)
$$

$$
\delta^l = \frac{\partial C}{\partial z^l} = \frac{\partial C}{\partial a^l} \odot \sigma^{'}(z^l)
$$

$$
\delta^l = ((w^{l+1})^T \cdot \delta^{l+1}) \odot \sigma^{'}(z^l)
$$

$$
\frac{\partial C}{\partial b_j^l} = \frac{\partial C}{\partial z_j^l} \cdot \frac{\partial z_j^l}{\partial b_j^l} = \frac{\partial C}{\partial z_j^l} \cdot 1 = \frac{\partial C}{\partial z_j^l}
$$

$$
\frac{\partial C}{\partial w_{jk}^l} 
= \frac{\partial C}{\partial z_j^l} \cdot \frac{\partial z_j^l}{\partial w_{jk}^l}
= \frac{\partial C}{\partial z_j^l} \cdot \frac{\partial (w_{jk}^l a_k^{l-1}+b_j^l)}{\partial w_{jk}^l}
= \frac{\partial C}{\partial z_j^l} \cdot a_k^{l-1} 
$$


二次（quadratic）代价函数

$$
C = \frac{1}{2n} \sum_{i=1}^n (y_i-a^L(x_i))^2
$$

交叉熵（cross entropy）代价函数

$$
C = - \frac{1}{n} \sum_{i=1}^n (y \ln a^L(x_i) + (1-y) \ln (1-a^L(x_i)))
$$

softmax:

$$
softmax(z_k) = \frac{e^{z_k}}{\sum_{i=1}^n e^{z_i}}
$$

$$
sigmoid(z) = \frac{1}{1+e^{-z}}
$$


Dropout

neurons saturated



sigmoid 不是non-zero center，即不对原点对称

https://blog.csdn.net/weixin_41417982/article/details/81437088

rectified linear unit




## Reference

- [神经网络的理解与实现](https://www.cnblogs.com/lliuye/p/9183914.html)



