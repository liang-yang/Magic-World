<!-- toc -->

# Image Feature Extraction

---

在 sklearn 中，对于平面图片，以二维数组存储。如果包含颜色信息，例如RGB格式使用3色通道，则以三维数组存储。

图像特征提取主要的工具类是 sklearn.feature_extraction.image，包括：

1. [sklearn.feature_extraction.image.extract_patches_2d](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.extract_patches_2d.html)：将某个图片划分为多个固定大小的小块，不过需要注意，划分的小块是存在重叠的。例如 4\*4 的图片，划分为2\*2尺寸的子块时，不仅仅只有4块，而是9块；
2. [sklearn.feature_extraction.image.reconstruct_from_patches_2d](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.reconstruct_from_patches_2d.html)：将图像的子块重新组合到一起；

{%ace edit=false, lang='java'%}
import numpy as np
from sklearn.feature_extraction import image

one_image = np.arange(4 * 4).reshape((4, 4))
print(one_image)
// [[ 0  1  2  3]
//  [ 4  5  6  7]
//  [ 8  9 10 11]
//  [12 13 14 15]]
patches = image.extract_patches_2d(one_image, (2, 2))
print(patches.shape)
// (9, 2, 2)
reconstructed = image.reconstruct_from_patches_2d(patches, (4, 4))
print(reconstructed)
// [[ 0.  1.  2.  3.]
//  [ 4.  5.  6.  7.]
//  [ 8.  9. 10. 11.]
//  [12. 13. 14. 15.]]
{%endace%}

另外，还有一个 [sklearn.feature_extraction.image.PatchExtractor](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.image.PatchExtractor.html) 类，原理和 extract\_patches\_2d 完全一致，只是 PatchExtractor 可以支持多图片，而且其实现为一个 estimator，因此可以放到 pipelines 中运行。

{%ace edit=false, lang='java'%}
five_images = np.arange(5 * 4 * 4 * 3).reshape(5, 4, 4, 3)
patches = image.PatchExtractor((2, 2)).transform(five_images)
print(patches.shape)
// (45, 2, 2, 3)
{%endace%}