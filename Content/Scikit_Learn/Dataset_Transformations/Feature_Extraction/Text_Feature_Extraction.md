<!-- toc -->

# Text Feature Extraction

---

scikit-learn提供了从文本内容中提取数字特征的最常见方法，即：

1. **tokenizing** ：对每个可能的词令牌分成字符串并赋予整数形的id，例如通过使用空格和标点符号作为令牌分隔符；
2. **counting** ：每个词令牌在文档中的出现次数；
3. **normalizing & weighting** ：对出现在大多数文档中的词令牌进行标准化和加权，使其重要性降低；

在该方案中，特征和样本定义如下：

- **特征**：每个单独的词令牌发生频率（归一化或不归零）被视为一个特征；
- **样本**：给定文档中所有的令牌频率向量被看做一个样本；

因此，文本的集合可被表示为矩阵形式，每行对应一条文本，每列对应每个文本中出现的词令牌(如单个词)。

通过这三个步骤将文本文档转换为特征向量的方式称作 “**Bag of Words**” 或 “**Bag of n-grams**”。 

由于此种方式处理后的矩阵很稀疏，因此会通过 scipy.sparse 包进行存储和处理。

## 1. Tokenizing And Counting

sklearn 通过类 [sklearn.feature_extraction.text.CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 实现 tokenizing 和 counting。

需要注意：

1. 默认的 tokenizing 要求单词长度大于2；
2. 我们可以通过函数 build_analyzer 得到执行 tokenizing 的函数；
3. 如果训练（fit）中不包含某word，那么转换（transform）时此word不会在 feature list 中；
4. scikit-learn 支持 N-grams 分词，由构造函数的 ngram\_range 参数控制。因此，如果需要保持词令之间的顺序，可以通过设置 ngram\_range 大于1来实现。另外，注意 ngram\_range 为数组；
5. 可通过参数 max\_df 和 min\_df 自动检测语料库内文档的词频生成 stop words ；

关键参数：

Parameters | Data-Type | Default | Comment | Note
:-:|:-:|:-:|:-:|:-:
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

示例代码：

{%ace edit=false, lang='java'%}
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(1, 2))
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
// ['and', 'and the', 'document', 'first', 'first document', 'is', 'is the', 'is this', 'one', 'second', 'second document', 'second second', 'the', 'the first', 'the second', 'the third', 'third', 'third one', 'this', 'this is', 'this the']
print(X.toarray())
// [[0 0 1 1 1 1 1 0 0 0 0 0 1 1 0 0 0 0 1 1 0]
// [0 0 1 0 0 1 1 0 0 2 1 1 1 0 1 0 0 0 1 1 0]
// [1 1 0 0 0 0 0 0 1 0 0 0 1 0 0 1 1 1 0 0 0]
// [0 0 1 1 1 1 0 1 0 0 0 0 1 1 0 0 0 0 1 0 1]]
{%endace%}

## 2. TF-IDF weighting

在一个文本语料库中，有一些单词将出现很多次但实际并没有什么意义，例如：the、a、an 等。因此，我们需要通过计算各个词令的权重值以确定其重要性，而 TF-IDF 就是最常用的方法。

我们令：
- $$term(t)$$ 表示单个词令
- $$document(d)$$ 表示当前文档
- $$documents(D)$$ 表示整个语料库

则：
- $$N(t,d)$$ 表示此词令在当前文档中出现的次数
- $$N(d)$$ 表示当前文章的总词数
- $$N(D)$$ 表示语料库中的文档总数
- $$N(D,t)$$ 表示语料库中包含当前词令的文档数量

有：
$$
TF(term\_frequency) = \frac{N(t,d)}{N(d)}
$$

$$
IDF(inverse\_document\_frequency) = \log \frac{N(D)}{N(D,t)}
$$

> 为了保证平滑，IDF 公式可能还会做一些处理，例如在分子、分母+1，或者最终结果+1，即 $$ IDF = \log (\frac{N(D)+1}{N(D,t)+1}) + 1$$。

$$
TF\_IDF = TF \times IDF
$$

最后，对 TF-IDF 正则化，即 $$ \vec{v} = \frac{\vec{v}}{|v|}$$，这就是最终的向量。

在 sklearn 中通过类 [sklearn.feature_extraction.text.TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) 进行 TF-IDF 处理：

{%ace edit=false, lang='java'%}
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer()
counts = [[3, 0, 1], [2, 0, 0], [3, 0, 0], [4, 0, 0], [3, 2, 0], [3, 0, 2]]
tfidf = transformer.fit_transform(counts)
print(transformer.idf_)
// [1.         2.25276297 1.84729786]
print(tfidf.toarray())
// [[0.85151335 0.         0.52433293]
//  [1.         0.         0.        ]
//  [1.         0.         0.        ]
//  [1.         0.         0.        ]
//  [0.55422893 0.83236428 0.        ]
//  [0.63035731 0.         0.77630514]]
{%endace%}

另外，sklearn 提供了 [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)，集成了 [sklearn.feature_extraction.text.CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 和 [sklearn.feature_extraction.text.TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) 两个类的功能。

## 3. 场景技巧

针对不同的场景，通过不同的技巧进行处理。

### 3.1. 词令相似性

英文中，存在拼写错误的场景，如 words 误写作 wprds。中文中，也存在错别字的情况。那么，我们怎样把 wprds 纠正回 words 呢？

类 [sklearn.feature_extraction.text.CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 的构造函数中存在一个 analyzer 参数，默认为 'word'，表示根据空格分词来进行向量化。 而如果设置 analyzer 为 char_wb，则表示在空格分词后再根据 N-grams 再次进行分词，如下所示：

{%ace edit=false, lang='java'%}
from sklearn.feature_extraction.text import CountVectorizer

ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2))
counts = ngram_vectorizer.fit_transform(['words', 'wprds'])
print(ngram_vectorizer.get_feature_names())
// [' w', 'ds', 'or', 'pr', 'rd', 's ', 'wo', 'wp']
print(counts.toarray())
// [[1 1 1 0 1 1 1 0]
//  [1 1 0 1 1 1 0 1]]
{%endace%}

可以看出，两个8维向量中有4维是相同的，具有一定的相似性。

除了 char\_wb，analyzer 还可设置为 char。这样就不会空格分词，而是直接 N-grams 分词。一般来说，使用 char\_wb 效果更好一些。

### 3.2. 大数据量文本库

由于文本特征提取时会将特征向量映射到内存中，因此在处理大数据量文本库时可能会导致内存不足。此种情况下，处理前可先借助 Feature Hash 的思想进行降维。

在 sklearn 中，通过类 [sklearn.feature_extraction.text.HashingVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html) 同步实现 降维 和 向量化。

{%ace edit=false, lang='java'%}
from sklearn.feature_extraction.text import HashingVectorizer

hv = HashingVectorizer(n_features=10)
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X = hv.transform(corpus)
print(X.toarray())
// [[ 0.          0.          0.          0.          0.          0.
//   -0.57735027  0.57735027 -0.57735027  0.        ]
//  [ 0.          0.          0.          0.          0.          0.81649658
//    0.          0.40824829 -0.40824829  0.        ]
//  [ 0.          0.5         0.          0.         -0.5        -0.5
//    0.          0.         -0.5         0.        ]
//  [ 0.          0.          0.          0.          0.          0.
//   -0.57735027  0.57735027 -0.57735027  0.        ]]
{%endace%}




