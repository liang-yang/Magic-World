# Datasets

---

[sklearn.datasets](https://scikit-learn.org/stable/datasets/index.html)

Datasets 提供 三 种数据集接口：Loaders、Fetchers 和 Generations。

- **Loaders**：加载 小数据量数据集，也称 Toy datasets；
- **Fetchers**：下载 并 加载 大数据量数据集，也称 Real world datasets；
- **Generations**：根据输入参数人为控制生成数据集；

他们都会返回：

- X: array[n_samples * n_features]
- y: array[n_samples]

对于 Loaders 和 Fetchers，还可以通过 **DESCR** 获取 特征列表。
