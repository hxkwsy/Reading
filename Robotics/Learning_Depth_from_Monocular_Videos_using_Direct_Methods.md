# Learning Depth from Monocular Videos using Direct Methods
[cvpr18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Learning_Depth_From_CVPR_2018_paper.pdf)
[git](https://github.com/MightyChaos/LKVOLearner)

## Introduction
1. 主要的区别 between stereo and monocular
   1. unknown camera pose between frames
   2. ambiguity in scale.
2. direct visual odometry [10, 8, 6, 2]
3. Direct Visual Odometry (DVO) [27]

## Learning depth estimation from videos
1. 从序列中预测深度和pose分为两步
   1. 计算深度图和相机位姿作为loss, $L_ap(D,p)$为appearence error, $L_{prior}(D)$为prior cost
   $$ D^* ,p^* =\arg\min L_ap(D,p), + L_{prior}(D)$$
   2. 学习深度预测方法
   $$\min_{\theta_d}L(f_d(I;\theta_d),D^* )$$
   3. 这种两步优化方法是sub-optimal的，应为第二步对第一步的优化不直接
2. end-to-end training objective,
   1. $$ \min_{\theta_d,p} L_{ap}(f_d(I;\theta_d),p), + L_{prior}(f_d(I;\theta_d)) $$
   2. 引入辅助的pose预测器$f_p$
   $$ \min_{\theta_d} L_{ap}(f_d(I;\theta_d),f_p(f_d(I;\theta_d))), + L_{prior}(f_d(I;\theta_d)) $$
3. 尺度模糊Scale ambiguity
   1. Any two inverse depth maps differing only in scale (with the pose scaled correspondingly) are equivalent in the projective space.
   任意两个仅尺度不同的depth maps，在对应的pose尺度下是等价的，即$L_{ap}$ is scale-invariant
   2. 尺度模糊导致1.2.2没有局部最小值，导致预测的深度值越来越小
   3. 解决方法：normalize CNN的输出, 即除以均值
   $$ \eta(d_i)=\frac{Nd_i}{\sum_{j=1}^Nd_i} $$

## Differentiable direct visual odometry
1. refrence image $I$ with $x_i$, Depth $D$ with $d_i$, source image $I'$ with $x'_ i$
$$x'_ i=W(x_i;p,d_i)=<R \tilde x_i+d_it>$$
其中R为旋转矩阵，t为平移向量, $\tilde x_i$为$x_i$的齐次坐标, $<\cdot>$映射3D点到图像平面$<[x,y,z]^T>=[x/z, y/z]^T$
2. direct visual odometry 的目标是最小化
