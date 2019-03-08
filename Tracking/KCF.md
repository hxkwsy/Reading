# High-speed tracking with kernelized correlation filters

[arXiv](https://arxiv.org/abs/1404.7584)
[CSDN](https://blog.csdn.net/dengheCSDN/article/details/78130290)
[KCF（核化相关滤波）跟踪公式推导笔记](https://blog.csdn.net/discoverer100/article/details/53835507)
[代码流程](https://blog.csdn.net/SmileAngel_520/article/details/79955273)

## Introduction
1. discriminative learning methods 被广泛应用于tracking
2. tracking可看做在线学习问题

## Building blocks
### Linear regression
1. 训练目标：找到一个映射$f(\bold z)=\bold w^T \bold z$，最小化预测X和目标y间的均方误差
$$ \min_\bold w\sum_i(f(\bold x_i)-y_i)^2+\lambda||\bold w||^2 $$
> $\bold x, \bold w$为列向量，表示样本和权重，$y$为标量，表示标签

矩阵形式
$$\min_\bold w||X \bold w- \bold y||^2+\lambda ||\bold w||^2 $$
> $X=\{\bold x_i\}^T$为矩阵，$\bold y$为标签列向量

求导等于0
$$2X^T(X\bold w-\bold y)+2\lambda\bold w=0 $$

线性求解
$$ \bold w=(X^TX+\lambda I)^{-1}X^T\bold y $$
傅里叶域的形式
$$ \bold w=(X^HX+\lambda I)^{-1}X^H\bold y $$
> $X^H=(X^* )^T$是共轭转置

### Cyclic shifts/matrices
1. 考虑一维单通道情况，$\bold x$的维度是$n\times 1$，作为base sample (positive example)。为了找到negative examples，利用排列矩阵建立一个 cyclic shift operator, 例如
$$P=\left[\begin{array}c
0 & 0 & 0 & \cdots & 1 \\
1 & 0 & 0 & \cdots & 0 \\
0 & 1 & 0 & \cdots & 0 \\
\vdots&\vdots& \ddots& \ddots& \vdots \\
0 & 0 & \cdots & 1 & 0
\end{array}\right]$$
> $P\bold x=[x_n,x_1,...,x_{n-1}]$, $P^u$可以产生更大的变换，$u<0$可以反向变换
> $n$次变换后可得到原信号，所以完全的偏移信号集合是$\{P^u\bold x|u=0,...,n-1\}$，集合的前一半可看做是正向偏移，后一半可看作是负向偏移

### Circulant matrices
1. Circulant matrices
$$X=\left[\begin{array}c
x_1 & x_2 & x_3 & \cdots & x_n \\
x_n & x_1 & x_2 & \cdots & x_{n-1} \\
x_{n-1} & x_n & x_1 & \cdots & x_{n-2} \\
\vdots&\vdots& \ddots& \ddots& \vdots \\
x_2 & x_3 & \cdots & x_n & x_1
\end{array}\right]$$
这个模式是完全由$\bold x$(第一行)确定的
2. Circulant matrices可以用离散傅里叶变换变成对角阵
$$ X=F diag(\hat\bold x)F^H $$
> $F$是一个常量矩阵$\mathcal F(\bold z)=\sqrt nF\bold z$，$\hat\bold x=\mathcal F(\bold x)$是DFT
> 这是一个特征分解的过程

### Putting it all together
$$X^HX=Fdiag(\hat\bold x^* )F^HFdiag(\hat\bold x )F^H=Fdiag(\hat\bold x^* )diag(\hat\bold x )F^H=Fdiag(\hat\bold x^* \odot \hat\bold x )F^H$$
> $F$是酉性矩阵, $\odot$是逐元素相乘

$$\hat\bold w=diag(\frac{\hat\bold x^* }{\hat\bold x^* \odot \hat\bold x+\lambda})\hat\bold y=\frac{\hat\bold x^* \odot\hat\bold y}{\hat\bold x^* \odot\hat\bold x+\lambda}$$

## Non-linear regression
### Kernel trick
利用kernel trick映射一个线性问题的输入到一个非线性特征空间$\phi(\bold x)$ 包含以下几部分
1. 用一个线性组合表达$\bold w=\sum_i\alpha_i\phi(\bold x_i)$, 从而优化$ \bm\alpha$而不是$\bold w$
2. 写成点积的形式$\phi^T(\bold  x)\phi(\bold  x')=k(\bold x,\bold x')$
样本的点积存在一个kernel matrix $K$中$K_{i,j}=k(x_i,x_j)$
计算复杂度随样本的增长而提升
$$f(\bold z)=\bold w^T\bold z=\sum_{i=1}^n\alpha_ik(\bold z,\bold x_i)$$

### Fast kernel regression
基于核的岭回归的解为
$$ \bm\alpha=(K+\lambda I)^{-1}\bold y $$
上式可被对角化
$$\hat\bm\alpha=\frac{\hat\bold y}{\hat\bold k^{\bold x\bold x}+\lambda}$$
$\bold k^{\bold xx}$为$K$的第一行，$\bold k^{\bold x\bold x'}$为核相关，其元素为
$$k_i^{\bold x\bold x'}=k(x',P^{i-1}\bold x)=\phi^T(\bold x')\phi(P^{i-1}\bold x)$$

### Fast detection
$$K^\bold z=C(\bold k^{\bold x\bold z})$$
是一个不对称的和矩阵，关于训练样本和所有的candidate pathes
$$f(\bold z)=(K^{\bold z})^T\bm\alpha$$
$f(\bold z)$是full detection response，经对角化后
$$\hat f(\bold z)=\hat\bold k^{\bold x\bold z}\odot \hat\bm\alpha$$
