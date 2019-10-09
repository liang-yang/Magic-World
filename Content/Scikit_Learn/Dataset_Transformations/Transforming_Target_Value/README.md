<!-- toc -->

# Transforming Target Value

---

see [Transforming the prediction target (y)](https://scikit-learn.org/stable/modules/preprocessing_targets.html)

除了可以转换特征向量，还可以转换结果值。

## 1. Label Binarization

[sklearn.preprocessing.LabelBinarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html) 将列向量转化为矩阵。

{%ace edit=false, lang='java'%}
from sklearn import preprocessing

lb = preprocessing.LabelBinarizer()
lb.fit([1, 2, 6, 4, 2])
print(lb.classes_)
// [1 2 4 6]
print(lb.transform([1, 6]))
// [[1 0 0 0]
//  [0 0 0 1]]
{%endace%}

## 2. Label Encoding

[sklearn.preprocessing.LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) 将列向量转化为标签化后的列向量。

{%ace edit=false, lang='java'%}
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit([1, 2, 2, 6])
print(le.classes_)
// [1 2 6]
print(le.transform([1, 1, 2, 6]))
// [0 0 1 2]
print(le.inverse_transform([0, 0, 1, 2]))
// [1 1 2 6]
{%endace%}

