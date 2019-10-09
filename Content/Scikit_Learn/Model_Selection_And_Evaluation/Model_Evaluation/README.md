<!-- toc -->

# Model Evaluation

---

see [Model evaluation: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html)

> Model Evaluation（模型评估） 使用的类及函数在 ? 包中。

模型评估主要有 三 种表现形式：

- **Estimator Score Method**

    Estimators 是算法模型，一般会自带的 score 函数作为默认的评估标准。具体的分析在各个Estimators的相关文档，并不在这个章节；
    
- **Metric Functions**

    在 [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) 包中，系统预定义了大量通用的评估指标，包括 Classification、Regression、Clustering 等多种模型；

- **Scoring Parameter**

    评估参数，主要供 交叉验证( Cross Validation) 和 网格搜索(Grid Search) 使用；
