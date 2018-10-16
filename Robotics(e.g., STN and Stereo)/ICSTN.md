# Inverse Compositional Spatial Transformer Networks

## Introduction
1. CNN如何保持空间形变不变形（tolerate spatial variations）
   1. spatial pooling layers: 破坏空间结构
   2. data augmentation techniques：实际上是增加数据
2. Lucas & Kanade (LK) [14]:
   1. a feed forward network of multiple alignment modules
   2. each alignment module contains a low-capacity predictor (typically linear) for predicting geometric distortion from relative image appearance
   3. followed by an image resampling/warp operation
   > 多个联合线性预测器来预测几个失真

   4. minimizing the sum of squared differences objective
   $$ \min_{\Delta p}||I(p+\Delta p)-\tau(0)||_ 2^2 $$
   > $I$为图像，$\Delta p$为估计的warp update

   5. Inverse Compositional algorithm (IC-LK)
   $$ \min_{\Delta p}||I(p)-\tau(\Delta p)||_ 2^2 $$

   6. Advantage of IC:
      1. its efficiency of computing the fixed steepest descent image $\frac{\partial \tau(0)}{\partial p}$ in the least-squares objective
      > it is evaluated on the static template image T at the identity warp $p = 0$ and remains constant across iterations, and thus so is the resulting linear regressor $R$

3.  Supervised Descent Method (SDM) [19]
4. more formally established link between LK and SDM [11]
5. STN
   1. 根据输入来扭曲图像
   $$ I_{out}(0)=I_{in}(p), p=f(I_{in}(0)) $$
   2. $f$为非线性方程，需要geometric predictor来预测warp parameters

## ICSTN
![ICSTN](./.assets/ICSTN.jpg)
1. compositional STNs (c-STNs)
    1. geometric transformation is also predicted from a geometric predictor
    2. warp parameters $p$ are kept track of, composed, and passed through the network instead of the warped images
    > 通过网络来整合扭曲参数，而不是直接扭曲图像

    3. updates to the warp can be iteratively predicted.
    4. This eliminates the boundary effect because pixel information outside the cropped image is also preserved until the final transformation
    > 消除了边缘效应，因为图像信息一直被保存，指代最后一次变换

    5. Mathematically
       1. $p=p[p_1, p_2, ..., p_6]$
       2. homogeneous coordinates
       $$M(p)=\left[\begin{array}c
       1+p_1 & p_2 & p_3 \\
       p_4 & 1+p_5 & p_6 \\
       0 & 0 & 1
       \end{array}\right]$$
       3. warp composition
       $$ M(p_{out})=M(\Delta p)\cdot M(p_{in}) $$
       4. derivative $\frac{\partial p_{out}}{\partial p_{in}}, \frac{\partial p_{out}}{\partial \Delta p}$
       > 微分的表达形式类似残差，所以 are insensitive to the vanishing gradient phenomenon given the predicted warp parameters $\Delta p$ is small

    6. 线性的cSTN等价于compositional LK algorithm
2. ICSTN
![icstn](./.assets/icstn_vqrg8p24t.jpg)
ICSTN就是迭代的cSTN

## Reference
[11] C.-H. Lin, R. Zhu, and S. Lucey. The conditional lucas &
kanade algorithm. In European Conference on Computer
Vision (ECCV), pages 793–808. Springer International Publishing, 2016.

[14] B. D. Lucas, T. Kanade, et al. An iterative image registration technique with an application to stereo vision. In IJCAI, volume 81, pages 674–679, 1981.

[19] X. Xiong and F. De la Torre. Supervised descent method
and its applications to face alignment. In Proceedings of the
IEEE conference on computer vision and pattern recognition, pages 532–539, 2013.
