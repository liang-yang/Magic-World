<!-- toc -->

# Dataset Loading Utilities

---

see [Dataset Loading Utilities](https://scikit-learn.org/stable/datasets/index.html)

> Dataset Loading Utilities 使用的类与函数均在 [sklearn.datasets](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets) 包中。

Dataset Loading Utilities 提供 三 种数据集接口：Loaders、Fetchers 和 Generations。

- **Loaders**：加载 小数据量数据集，也称 Toy Datasets；
- **Fetchers**：下载 并 加载 大数据量数据集，也称 Real World Datasets；
- **Generations**：根据输入参数人为控制统计属性生成数据集；

他们都会返回：

- X: array[n_samples, n_features]
- y: array[n_samples]

对于 Loaders 和 Fetchers，还可以通过 **DESCR** 获取 特征列表。