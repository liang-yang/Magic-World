<!-- toc -->

# Cross-Validation

---

see [Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)

> Cross-Validation 使用的类及函数均在 [sklearn.model_selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection) 包中。

关于 Cross-Validation(交叉验证)，首先**必须要理解 训练集、验证集、测试集 各自的作用，尤其是 验证集 和 测试集 的差别**：

1. **训练集**：

    - 狭义的 训练集 是指用于训练出模型集合的样本集（注意，这里训练得到的是模型集合，即不只是单个模型）；
    - 广义的 训练集 包括 狭义训练集 和 验证集；
    
2. **验证集**：

    - 验证集 是用于筛选比较 狭义训练集 训练出的模型集合，输出一个最优的最终模型；

3. **测试集**：

    - 测试集 是用于评估最终模型的泛化能力，不用作模型的选择和修正；

明白了 测试集 仅用作模型泛化能力的评估，而不用作模型的选择和修正，就理解了验证集的必要性。  
但是，当样本量较少时，这样会导致 训练集 的数据量更少，进而影响模型的运算（理论上，训练样本越多，信息越充分，模型越精确）。  
另外，随机划分 训练集 和 验证集，会使得 模型 非常依赖划分的随机性：划分的好模型就好，划分的不好模型就可能不好。

因此，引入了 Cross-Validation（CV，交叉验证），目的是将样本集尽可能多的用作训练，同时也避免划分的随机性对模型的影响。
