# Geometric loss functions for camera pose regression with deep learning
[arXiv](https://arxiv.org/abs/1704.00390)
[cvpr](http://openaccess.thecvf.com/content_cvpr_2017/papers/Kendall_Geometric_Loss_Functions_CVPR_2017_paper.pdf)

## Method
1. Quaternions
   1. arbitrary four dimensional values are easily mapped to legitimate rotations by normalizing them to unit length. 任何一个四元素都是合法的
   2. Quaternions are a continuous and smooth representation of rotation.
   3. 缺点: they have two mappings for each rotation, one on each hemisphere. i.e., $q=-q$, 有必要消除这个影响

2. learn a scale
   1. Homoscedastic uncertainty
$$ L = L_{pos}\hat\sigma_{pos}^{-2}+\log\sigma_{pos}^{2}+L_{ori}\hat\sigma_{ori}^{-2}+\log\sigma_{ori}^{2} $$
     optimise the homoscedastic uncertainties, i.e., $\sigma_{pos}^{2},\sigma_{ori}^{2}$, through backpropagation with respect to the loss function.
   2. 更稳定的做法
   $$L = L_{pos}\exp( -\hat s_{pos})+ \hat s_{pos} + L_{ori}\exp(-\hat s_{ori})+ \hat s_{ori} $$
   初始化：$\hat s_{pos}=0, \hat s_{ori}=-3$

3. Reprojection error [14]
   1. Reprojection error is given by the residual between 3-D points in the scene projected onto a 2-D image plane using the ground truth and predicted camera pose.  It therefore **converts rotation and translation quantities into image coordinates**.
   2. maps a 3-D point to 2-D image coordinates
   $\pi(x,q,g)\to (u,v)$
   > $g$为3D points, $x$为local，$q$为四元数, $(u,v)$为2D图像坐标

   $ (u', v', w')=K(Rg+x), (u,v)=(u'/w', v'/w') $
   > $K$是内参数矩阵,可被设为单位阵, $R$是旋转矩阵, $q_{4\times1}\to R_{3\times3}$

   3. loss
   $$L=\frac{1}{|G'|}\sum_{g_i\in G'}||\pi(x,q,g_i)-\pi(\hat x,\hat q, g_i)||_ \gamma$$
   > $G'$是图像中可见的**三维点(额外的信息)**
   > if the scene is very far away, then rotation is more significant than translation and vice versa.

4. Regression norm
