<!-- toc -->

# Classification Metrics

---

> see [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

我们以 $$ n $$ 分类举例，样本集合为 $$ S $$，分类类别集合为 $$ C=\{c_i, c_2, ... c_{n-1}, c_n\} $$。

对于所有分类模型（不仅仅是二分类），其中每一个分类类别的预测结果均可以分为四类：

- **TP** ———— True Positive
- **FP** ———— Flase Positive
- **FN** ———— False Negative
- **TN** ———— True Negative

— |真实结果：$$c_i$$|真实结果：$$not \space c_i$$
:-:|:-:|:-:
**预测结果：$$c_i$$** | $$TP_{c_i}$$ | $$FP_{c_i}$$
**预测结果：$$not \space c_i$$** | $$FN_{c_i}$$ | $$TN_{c_i}$$

