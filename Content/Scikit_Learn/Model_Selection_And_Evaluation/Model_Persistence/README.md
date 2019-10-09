<!-- toc -->

# Model Persistence

---

see [Model Persistence](https://scikit-learn.org/stable/modules/model_persistence.html)

> Model Persistence 使用到 pickle 和 joblib 模块。

在训练完 scikit-learn 模型之后，最好将模型持久化以备将来使用，而无需重新训练（不同版本的scikit-learn模型不能混合加载）。

最简单的，可使用 Python 内置的 [pickle](https://docs.python.org/3.7/library/pickle.html) 模块持久化模型（既可以二进制形式在内存中传递，也可以文件形式在硬盘存储）。另外，也可通过 [joblib](https://joblib.readthedocs.io/en/latest/persistence.html)，这对于内部带有 numpy 数组的对象来说更为高效。
 
{%ace edit=true, lang='java'%}
from sklearn import svm
from sklearn import datasets
import pickle
import joblib

clf1 = svm.SVC(gamma='scale')
iris = datasets.load_iris()
clf1.fit(iris.data, iris.target)

model_memory = pickle.dumps(clf1)
clf2 = pickle.loads(model_memory)

with open('model_disk.pickle', 'wb') as fw:
    pickle.dump(clf1, fw)
with open('model_disk.pickle', 'rb') as fr:
    clf3 = pickle.load(fr)

joblib.dump(clf1, 'model_disk.joblib')
clf4 = joblib.load('model_disk.joblib')

print(clf1)
print(clf2)
print(clf3)
print(clf4)

// 原始模型
// SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
//     decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
//     max_iter=-1, probability=False, random_state=None, shrinking=True,
//     tol=0.001, verbose=False)

// pickle模块 内存转换后的模型
// SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
//     decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
//     max_iter=-1, probability=False, random_state=None, shrinking=True,
//     tol=0.001, verbose=False)

// pickle模块 文件转换后的模型
// SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
//     decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
//     max_iter=-1, probability=False, random_state=None, shrinking=True,
//     tol=0.001, verbose=False)

// joblib模块 文件转换后的模型
// SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
//     decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
//     max_iter=-1, probability=False, random_state=None, shrinking=True,
//     tol=0.001, verbose=False)
{%endace%}
