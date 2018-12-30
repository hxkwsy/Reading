# Continuous Conditional Random Fields for Efficient Regression in Large Fully Connected Graphs
[aaai13](http://www.dabi.temple.edu/~zoran/papers/KostaAAAI13.pdf)

## Introduction
1. CNN回归基于独立同分布假设 identically distributed (IID)，但实际中不满足
The IID assumption is often violated in structured data, where examples exhibit sequential, temporal, spatial, spatio-temporal, or some other dependencies.
2. Markov Random Fields (MRF)
> Solberg, A H S, T. T., and Jain, A. K. 1996. A Markov Random Field Model for Classification of Multisource Satellite Imagery. IEEE Transactions on Geoscience and Remote Sensing 34(1):100–113.

3. Conditional Random Fields (CRF)
> Lafferty, J, M. A., and Pereira, F. 2001. Conditional random fields: Probabilistic Models for Segmenting and Labeling Sequence Data. In Proceedings International Conference on Machine Learning.

4. Continuous CRF (CCRF)
> Qin, T.; Liu, T.; Zhang, X.; Wang, D.; and Li, H. 2008. Global Ranking Using Continuous Conditional Random Fields. Neural Information Processing Systems.

该文章用于 document retrieval，其余应用还有
   + denoising
   > Tappen, M F, L. C.; Adelson, E. H.; and Freeman, W. T. 2007. Learning Gaussian Conditional Random Fields for Low-Level Vision. IEEE Conference on Computer Vision and Pattern Recognition.

   + remote sensing
   > Radosavljevic, V.; Vucetic, S.; and Obradovic, Z. 2010. Continuous conditional random fields for regression in remote sensing. In Proceedings of the 2010 conference on ECAI 2010: 19th European Conference on Artificial Intelligence, 809–814. Amsterdam, The Netherlands, The Netherlands: IOS Press.

5. mean field theory
> Koller, D., and Friedman, N. 2009. Probabilistic Graphical Models: Principles and Techniques. MIT Press.

## Fully Connected Continuous Conditional Random Fields
1. CCRF用于建模条件分布$P(y|X),y=(y_1,...,y_N)$
$$\begin{array}l P(y|X)=\frac{1}{Z(X,\alpha,\beta)}\exp(\phi(y,X,\alpha,\beta)) \\[10pt]
\phi(y,X,\alpha,\beta)=\sum_{i=1}^NA(\alpha,y_i,X)+\sum_{i\sim j}I(\beta,y_i,y_j,X) \\[10pt]
Z(X,\alpha,\beta)=\int_y\exp(\phi(y,X,\alpha,\beta))dy
\end{array}$$
   1. Z必须可积分integrable，整个模型才可行
   2. $A(\alpha,y_i,X)$为association potential，$\alpha$为$K$维参数
   3. $I(\beta,y_i,y_j,X)$为interaction potential，建模相互的关系，$\beta$为$L$维参数
   4. $A,I$由一系列特征方程$f,g$线性组合
   $$\begin{array}l
   A(\alpha,y_i,X)=\sum_{k=1}^K\alpha_kf_k(y_i,X) \\[10pt]
   I(\beta,y_i,y_j,X)=\sum_{l=1}^L\beta_lg_l(y_i,y_j,X)
   \end{array}$$
2. 假设有$K$个非结构化的模型$R_k(X)$可以预测$y_i$，可以构建二次特征方程for $A$和$I$
$$\begin{array}l
f_k(y_i,X)=-(y_i-R_k(X))^2,k=1,...,K \\[10pt]
g_l(y_i,y_j,X)=-k_l(p_i^{(l)},p_j^{(l)})(y_i-y_j)^2,l=1,..,L
\end{array}$$
其中$k_l>0$用于测量任意特征空间中特征相量$p$的相似度
## Approximation of Fully-connected CCRF Using Fast Gaussian Filtering
1. 精确的推理需要$O(N^3)$，由于计算矩阵逆
2. 利用mean field来减少计算量：用$Q(y|X)=\Pi_{i=1}^NQ_i(y_i|X)$ (a product of independent marginals)估计$P(y|X)$
3. 根据mean field理论，最好的$Q$应使$P,Q$的KL散度最小
$$\log(Q_i(y_i|X))=E_{j\neq i}[\log P(y|X)]+const$$
其中，$E_{j\neq i}$表示关于$Q$的概率分布，对于$y_j,j\neq i$
4. 根据以上公式
$$\log(Q_i(y_i|X))=-\sum_{k=1}^K\alpha_k(y_i^2-2y_iR_k(X))-2\sum_{l=1}^L\beta_l\sum_{j\neq i}k_l(p_i^{(l)},p_j^{(l)})(y_i^2-2y_iE[y_i])+const$$
5. 由于$\log Q_i(y_i|X)$是关于$y_i$二次形式，可以写成高斯分布，其均值和方差为
$$
\mu_i=\frac{\sum_{k=1}^K\alpha_kR_k(X)+2\sum_{l=1}^L\beta_l\sum_{j\neq i}k_l(p_i^{(l)},p_j^{(l)})\mu_j}{\sum_{k=1}^K\alpha_k+2\sum_{l=1}^L\beta_l\sum_{j\neq i}k_l(p_i^{(l)},p_j^{(l)})}
$$
$$\sigma^2_i=\frac{1}{2(\sum_{k=1}^K\alpha_k+2\sum_{l=1}^L\beta_l\sum_{j\neq i}k_l(p_i^{(l)},p_j^{(l)}))}
$$
6. 进一步减小计算量
$$\begin{array}l
\sum_{j\neq i}k_l(p_i^{(l)},p_j^{(l)})\mu_j=\sum_{j}k_l(p_i^{(l)},p_j^{(l)})\mu_j-\mu_i \\
\sum_{j\neq i}k_l(p_i^{(l)},p_j^{(l)})=\sum_{j}k_l(p_i^{(l)},p_j^{(l)})\mu_j-1
\end{array}$$
```

```
