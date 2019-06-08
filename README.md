# 相关论文
## 1. 图像块描述子学习（旨在替代传统的手工描述子，在已检测特征点的情况下得描述子，除LIFT是检测与描述一起之外）
[1]Simo-Serra, Edgar, et al. "Discriminative learning of deep convolutional feature point descriptors." #Proceedings of the IEEE International Conference on Computer Vision. 2015.
[2] Xufeng Han, T. Leung, Y. Jia, R. Sukthankar, and A. C. Berg. Matchnet: Unifying feature and metric learning for patch-based matching. In Conference on Computer Vision and Pattern Recognition (CVPR), pages 3279–3286, 2015.
[3] Sergey Zagoruyko and Nikos Komodakis. Learning to compare image patches via convolutional neural networks. In Conference on Computer Vision and Pattern Recognition (CVPR), 2015
[4]Yang, Tsun-Yi, et al. "Deepcd: Learning deep complementary descriptors for patch representations." Proceedings of the IEEE International Conference on Computer Vision. 2017.
[5]Mishchuk, Anastasiia, et al. "Working hard to know your neighbor's margins: Local descriptor learning loss." Advances in Neural Information Processing Systems. 2017.
[6]Tian, Yurun, Bin Fan, and Fuchao Wu. "L2-net: Deep learning of discriminative patch descriptor in euclidean space." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.
[7]K. M. Yi, E. Trulls, V. Lepetit, and P. Fua. LIFT: LearnedInvariant Feature Transform. InProc. ECCV, 2016.

## 2. 图像特征描述子学习（包括全局和局部描述子，主要用于图像检索等领域，直接得图像特征描述子）
[1]R. Arandjelovi ́c, P. Gronat, A. Torii, T. Pajdla, and J. Sivic.NetVLAD: CNN Architecture for Weakly Supervised PlaceRecognition. InProc. CVPR, 2016.
[2]F. Radenovi ́c, G. Tolias, and O. Chum. CNN Image RetrievalLearns from BoW: Unsupervised Fine-Tuning with Hard Ex-amples. InProc. ECCV, 2016. 
[3]A. Gordo, J. Almazan, J. Revaud, and D. Larlus. Deep ImageRetrieval: Learning Global Representations for Image Search.InProc. ECCV, 2016. 
[4]Yue-Hei Ng, Joe, Fan Yang, and Larry S. Davis. "Exploiting local features from deep networks for image retrieval." Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2015.
[5]Noh, Hyeonwoo, et al. "Large-scale image retrieval with attentive deep local features." Proceedings of the IEEE International Conference on Computer Vision. 2017.

## 3. 特征描述子聚合（利用CNN实现的的BOV,Fisher Vector,VLAD或改进版本）
[1]Cimpoi M, Maji S, Vedaldi A. Deep filter banks for texture recognition and segmentation[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 3828-3836.
[2]Arandjelovic R, Gronat P, Torii A, et al. NetVLAD: CNN architecture for weakly supervised place recognition[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 5297-5307.
[3]Lin, Tsung-Yu, Aruni RoyChowdhury, and Subhransu Maji. "Bilinear cnn models for fine-grained visual recognition." Proceedings of the IEEE international conference on computer vision. 2015.

# 论文解读
## 1. DELF
>  "Large-scale image retrieval with attentive deep local features."
> https://arxiv.org/pdf/1612.06321.pdf
1. Introduction
在过去的几十年里，图像检索系统取得了巨大的进步，从手工制作的特征和索引算法到最近基于卷积神经网络(CNNs)的全局描述符学习方法。全局描述符缺乏在图像之间找到匹配的块的能力。因此，在存在遮挡和背景杂乱的情况下，根据局部块来检索相应图像是比较困难的。
在最近的一个趋势中，提出了基于cnn的局部特征用于图像块匹配[1中相关论文]。然而，这些技术并没有专门针对图像检索进行优化，因为它们缺乏检测语义上有意义的特征的能力，并且在实践中显示出有限的准确性。

文章的主要目标是开发一个基于新的基于cnn的特征描述符的大规模图像检索系统。首先引入了一个新的大规模的dataset, Google-Landmarks，它包含了超过1M landmark的图像，来自近13K个独特的landmark。该数据集覆盖了世界上的广泛领域，因此比现有数据集更加多样化和全面。查询集由多出的10万幅具有不同特征的图像组成;特别是，我们在数据库中包含了没有匹配的图像，这使得我们的数据集更具挑战性。然后，提出了一种基于cnn的带注意力集中机制的局部特征，它只使用图像级的类标签进行弱监督训练，不需要对象级和图像块的注释，这个新的特性描述符称为DELF。

2. Image Retrieval with DELF
文中的大规模检索系统可以分解为四个主要模块:(i)密集局部特征提取，(ii)关键点选择，(iii)降维和(iv)索引和检索。
2.1 密集局部特征提取
文中采用全卷积网络提取图像的稠密特征，并使用分类损失进行训练。具体的FCN来自ResNet50，使用conv4_x卷积块的输出作为输出。为了解决尺度变化问题，文中构建了图像金字塔，对每一个尺度独立地使用FCN。得到的特征图被认为是由局部描述符组成的密集网格。将接受域中心的像素坐标作为特征位置。图像在原始尺度下的接受域大小为291×291，利用图像金字塔，得到了描述不同大小图像区域的特征。
2.2 关键点选择 Attention-based Keypoint Selection
文中设计了一种有效提取特征子集的技术，而不是直接使用密集提取的特征进行图像检索。先训练得到图像描述子，然后固定描述子再去学习attention机制中的score function，在训练的时候对图像进行随机的scaling等，希望注意力机制能抓住不同情况下的共有特征。注意的是，keypoint selection是在描述符提取之后进行的，这与现有技术不同。
2.3 PCA降维 Dimension Reduction
首先对所选特征进行了归一化处理，通过主成分分析将特征的维度降至40，在紧凑性和判别性之间取得了良好的平衡。最后，这些特性再一次被归一化。
2.4 索引和检索：Image Retrieval System（涉及到了很多检索常用的技术）
最近邻搜索，D-tree和ProductQuantization (PQ)，k-means聚类，Locally Optimized Product Quantizer，RANSAC几何验证等
![avatar]（https://img-blog.csdn.net/2018060622472677)

**个人理解**
整个过程：利用resnet提取到多个特征图后，特征图的每个空间位置的值可以映射回原始图像的某一块区域，取该区域的中心为特征点的位置，而特征图对应点的值为特征描述子（因为有多个特征图，所以是多维的），attention机制的作用是区分这些特征点的重要程度，如果某个特征点的分数不够，可以选择舍去该特征点。而降维则需要多个这样的特征描述子来训练降维矩阵。

