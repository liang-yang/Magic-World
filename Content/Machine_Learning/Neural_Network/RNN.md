<!-- toc -->

# RNN

RNN，Recurrent Neural Network，循环神经网络，在自然语言处理领域应用广泛。

## 语言模型

RNN 最早作为语言模型来建模，即在给定前文的基础上预测后面的词，如下例。

> 我 昨天 上学 迟到 了，老师 批评 了 \_\_\_\_。

在 RNN 之前，语言模型主要是采用 N-Gram，即假设一个词出现的概率只与前面 N 个词相关。 N-Gram 的模型大小与 N 的大小指数相关，N 不能设太大。 而 RNN 理论上可以往前看（往后看）任意多个词。


在 DNN 中，输入数据之间互相独立，但在 RNN 中，输入的数据具有时间上的先后次序，形成了一个序列。

![](https://tva1.sinaimg.cn/large/006y8mN6gy1g9eqa5opmnj30m308vmxa.jpg)

$$
s_t = \sigma(Ux_t + Ws_{t-1})
$$

> $$x$$ 代表了输入层，$$s$$ 是隐藏层，$$o$$ 代表了输出层。$$U, V, W$$ 分别是网络中的参数，$$\sigma$$ 表示激活函数。其中，$$s$$ 所代表的隐藏状态除了用来传递给输出层，又经过图中的箭头循环回来被再次使用。    
将这一循环结构展开就得到了图中右侧的结构：$$x_{t-1}$$ 到 $$x_{t+1}$$ 代表了不同时刻依次输入的数据，每一时刻的输入都会得到一个相应的隐藏状态 $$s$$，该时刻的隐藏状态除了用于产生该时刻的输出，也参与了下一时刻隐藏状态的计算。

将上式循环递归：

$$
s_t = \sigma(Ux_t + Ws_{t-1}) = \sigma(Ux_t + W \sigma(Ux_{t-1} + Ws_{t-2})) = \sigma(Ux_t + W \sigma(Ux_{t-1} + W \sigma(Ux_{t-2} + Ws_{t-3}))) = \cdots
$$

可以看出 RNN 的结果受到前面所有输入值的影响，也就是 RNN 可以往前看任意多个词。


循环神经网络的训练算法：BPTT(Backpropagation Through Time)



一段文字是一串字符组成的一个序列，一段视频是一串静止的图片组成的序列，一条K线是一串数据组成的序列



## Reference    

- [Recurrent Neural Networks Tutorial, Part 1 – Introduction to RNNs](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
- [循环神经网络](https://zybuluo.com/hanbingtao/note/541458)