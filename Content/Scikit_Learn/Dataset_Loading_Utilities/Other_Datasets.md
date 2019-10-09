<!-- toc -->

# Other Datasets

---

see [Other Datasets](https://scikit-learn.org/stable/datasets/index.html#loading-other-datasets)

## 1. sample images

Scikit-learn 提供两张2D的JPEG图片：'china.jpg', 'flower.jpg'。

[sklearn.datasets.load_sample_image](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_sample_image.html)

[sklearn.datasets.load_sample_images](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_sample_images.html)

{%ace edit=true, lang='java'%}
from sklearn.datasets import load_sample_image

china = load_sample_image('china.jpg')

print("dtype:", china.dtype)
// dtype: uint8
print("shape:", china.shape)
// shape: (427, 640, 3)
{%endace%}

需要注意，返回的数据是 uint8 格式，即无符号8字节int型，范围为0~255。如果有些场景应用需要转换为0~1，需要除以255。

## 2. svmlight or libsvm format

svmlight/libsvm 格式的数据集，如下所示：

    <label> <feature-id>:<feature-value> <feature-id>:<feature-value> ...

可通过如下方式加载：

[datasets.load_svmlight_file](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html)

[datasets.load_svmlight_files](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_files.html)

{%ace edit=true, lang='java'%}
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import load_svmlight_files

X_train, y_train = load_svmlight_file("/path/to/train_dataset.txt")
X_train, y_train, X_test, y_test = load_svmlight_files(("/path/to/train_dataset.txt", "/path/to/test_dataset.txt"))
{%endace%}

## 3. openml.org repository

openml.org 是一个公共的机器学习的数据库，我们可以通过 [sklearn.datasets.fetch_openml](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html) 从上面下载数据：

{%ace edit=true, lang='java'%}
from sklearn.datasets import fetch_openml

mice = fetch_openml(name='miceprotein', version=4)

print(mice.data.shape)
print(mice.target.shape)
print(np.unique(mice.target))
{%endace%}

## 4. external datasets

scikit-learn 支持 numpy array、scipy sparse matrices、pandas DataFrame 格式的数据。

其他的数据，可通过：

1. [pandas.io](https://pandas.pydata.org/pandas-docs/stable/io.html) 可读取 CSV, Excel, JSON, SQL 文件，转化为 scikit-learn 可使用的 DataFrame 格式；
2. [scipy.io](https://docs.scipy.org/doc/scipy/reference/io.html) 专用于 .mat、.arff 等二进制格式文件，此类数据常用于科学计算；
3. [numpy/routines.io](https://docs.scipy.org/doc/numpy/reference/routines.io.html) 读取数据转化为 numpy arrays；

建议在存储数据时通过 HDF5 等格式以减少加载时间。
