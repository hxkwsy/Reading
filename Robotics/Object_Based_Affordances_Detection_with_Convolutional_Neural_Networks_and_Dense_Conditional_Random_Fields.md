# Object-Based Affordances Detection with Convolutional Neural Networks and Dense Conditional Random Fields
[iros](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8206484)

## CRF
1. CRF for segmentation task [21] [22]
2. Energy function of dense CRF
$$
\begin{array}l E(x|P)=\sum_p\theta_p(x_p)+\sum_{p,q}\psi_{p,q}(x_p,x_q) \\[10pt]
\psi_{p,q}(x_p,x_q)=\mu(x_p,x_q)\sum_{m=1}^Mw^mk^m(f_p,f_q) \\[10pt]
k(f_p,f_q)=w_1\exp(-\frac{|p_p-p_q|^2}{2\sigma_\alpha^2}-\frac{|I_p-I_q|^2}{2\sigma_\beta^2})+w_2\exp(-\frac{|p_p-p_q|^2}{2\sigma_\gamma^2})
\end{array}
$$
> $\theta_p(x_p)$: cost of assigning label $x_p$ to pixel $p$; the output of the last layer from the affordance network since this layer produces a probability map for each affordance class. $p$点分类为$x_p$的代价，即网络最后一层的输出(这个输出就是一个像素点分类的概率map)
> $\psi_{p,q}(x_p,x_q)$: models the relationship among neighborhood pixels and penalizes inconsistent labeling. 建模相邻像素的关系，惩罚不一致的分类标签，可表达为weighted Gaussians
> $k^m, m=1,...,M$: a Gaussian kernel based on the features f of the associated pixels, and has the weights $w^m$
> $\mu(x_p,x_q)$ represents label compatibility and is $1$ if $x_p\neq x_q$, otherwise $0$.
> $p_p,p_q$代表位置，$I_p,I_q$代表color

3. Goal: minimize the CRF energy $E(x|P)$. Since the
dense CRF has billion edges and the exact minimization is intractable, we use the mean-field algorithm [26] to efficiently approximate the energy function.
