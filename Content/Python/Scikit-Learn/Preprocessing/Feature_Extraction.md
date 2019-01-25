<!-- toc -->

# Feature Extraction

---

[Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html) 主要是从 text 和 image 中提取特征。

## 1. Loading features from dicts

Python 原生的 dict 数据结构，虽然不太快，但具有简单、稀疏、可读等特点，常用作数据存储。但是，在 scikit-learn 使用时，我们需要通过 类 [DictVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html) 将 dict 转化为 numpy.array 或 SciPy.CSR 格式。

[DictVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html) 将无序的离散分类特征（字符串）转化为 one-of-K（one-hot）编码，数值型特征保持不变。


Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
dtype | callable | numpy.float64 | 转换后array或矩阵的元素类型 | -
separator | string | '=' | one-hot 编码时使用的属性分隔符 | -
sparse | boolean | True | 是否转化为 稀疏矩阵 | -
sort | boolean | True | 转化时 特征名称 是否排序 | -


Attributes | Data-Type | Comment | Note
:-:|:-:|:-:|:-:
vocabulary_ | dict | 特征名称的字典，value为特征排序 | -
feature_names_ | list | 特征名称的数组 | -


Methods | Parameters | Returns | Comment | Note
:-:|:-:|:-:|:-:|:-:
fit | X（特征的字典）| - | 训练特征名称 | -
transform | X（特征的字典）| Xa（转化后的特征向量）| 将特征转化为array或CSR | -
fit_transform | X（特征的字典）| Xa（转化后的特征向量）| 训练并转化特征向量 | -
get_feature_names | - | 特征名称的数组 | - | -
inverse_transform | X（样本的特征矩阵）| D（样本的特征映射）| 将数组或稀疏矩阵X转换回特征映射 | -
restrict | support | - | 对支持使用特征选择的模型进行特征限制，例如只选择前几个特征 | -


{%ace edit=true, lang='python'%}
from sklearn.feature_extraction import DictVectorizer

measurements = [{'city':'Beijing','country':'CN','temperature':33.},{'city':'London','country':'UK','temperature':12.},{'city':'San Fransisco','country':'USA','temperature':18.}]
vec = DictVectorizer()
measurements_vec = vec.fit_transform(measurements)
print(vec.feature_names_)
print(vec.vocabulary_)

print(measurements_vec)
print(measurements_vec.toarray())
{%endace%}

> 通过示例可以看出，多个离散分类特征（字符串）的 one-hot 编码的特征会加到一起，并不会做笛卡尔积。


## 2. Feature hashing

Feature hashing 是降维的一种手段，将 高维度特征向量 转化为 低维度特征向量，主要通过类 [FeatureHasher](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html) 实现。


Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
n_features | integer | 1048576 | 输出矩阵的特征列数量 | 此值太小较容易引起特征碰撞
input_type | string | 'dict' | 输入的数据类型 | -
dtype | numpy type | np.float64 | 特征值的类型，默认为浮点型 | -
alternate_sign | boolean | True | 不同特征hash到同一特征时是否交替变化正负符号，以抵消hash碰撞的影响 | -


Methods | Parameters | Returns | Comment | Note
:-:|:-:|:-:|:-:|:-:
transform | raw_X（将被hash的特征集合）| X（hash后的系数矩阵）| 特征hash | -


{%ace edit=true, lang='python'%}
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import FeatureHasher

measurements = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]

vec = DictVectorizer()
measurements_vec = vec.fit_transform(measurements)
print(measurements_vec)
print(measurements_vec.toarray())

hash = FeatureHasher(n_features=10)
measurements_hash = hash.transform(measurements)
print(measurements_hash)
print(measurements_hash.toarray())
{%endace%}


## 3. Text feature extraction

scikit-learn 提供如下方法将文本转化为数字化特征向量：

- **tokenizing** 
- **counting**
- **normalizing & weighting**

通过这三个步骤处理文本的方式称作 “**Bag of Words**”。

由于此种方式处理后的矩阵很稀疏，因此会通过 scipy.sparse 包进行存储和处理。

### 3.1 tokenizing & counting

通过类 [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 实现 tokenizing 和 counting。需要注意：

1. 默认的 tokenizing 要求单词长度大于2；
2. 我们可以通过函数 build_analyzer 得到执行 tokenizing 的函数；
3. 如果训练（fit）中不包含某word，那么转换（transform）时此word不会在feature list中；
4. scikit-lear 支持 N-grams 分词，由构造函数的 ngram_range 参数控制；
5. stop word 可通过语料库内文档的词频自动检测；


Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
input | {‘filename’, ‘file’, ‘content’} | ’content’ | 入参类型 | -
encoding | string | ‘utf-8’ | 编解码格式 | -
decode_error | {‘strict’, ‘ignore’, ‘replace’} | ‘strict’ | 编解码错误的处理方式 | -
strip_accents | {‘ascii’, ‘unicode’, None} | None | - | -
lowercase | boolean | True | 是否转化为小写字母 | -
preprocessor | callable | None | transformation模块 | -
tokenizer | callable | None | 分词模块 | -
stop_words | string {‘english’}, list | None | ‘english’加载既定词汇表，list则为stop词汇表 | -
token_pattern | string | ’(?u)\b\w\w+\b’ | tokenizing时的正则表达式，默认多个数字字母字符并通过标点符号分隔 | -
ngram_range | tuple (min_n, max_n) | (1, 1) | N-grams 分词中N的范围 | -
analyzer | {‘word’, ‘char’, ‘char_wb’} | ’word’ | ‘word’表示分出来的单词, ‘char’表示根据N-grams分出来的字符串, ‘char_wb’表示N-grams分词时不会跨边界 | -
max_df | float, int | 1.0 | 构建词汇表时，忽略文档词频高于此阈值的词汇。浮点型表示文档比例，整型表示绝对数量 | -
min_df | float, int | 1 | 构建词汇表时，忽略文档词频低于此阈值的词汇。浮点型表示文档比例，整型表示绝对数量 | -
max_features | int | None | 取语料库中词频排序的 top max_features 形成词汇表 | -
vocabulary | Mapping | None | 人为指定的词汇表 | -
binary | boolean | False | 如果为True，则所有非零的特征值均为1 | -
dtype | type | numpy.int64’ | transform() 和 fit_transform() 返回的数据格式 | -


Attributes | Data-Type | Comment | Note
:-:|:-:|:-:|:-:
vocabulary_ | dict | 特征列的映射，包含序列值 | -
stop\_words_ | set | 由于 max_df、min_df、max_features 等配置形成的 stop words | -


Methods | Parameters | Returns | Comment | Note
:-:|:-:|:-:|:-:|:-:
- | - | - | - | -
- | - | - | - | -
- | - | - | - | -
- | - | - | - | -
- | - | - | - | -


## 4. Image feature extraction
















