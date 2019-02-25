[TOC]

[面试](https://www.zhihu.com/question/23259302/answer/527513387?utm_source=qq&utm_medium=social&utm_oi=638711330211762176)
[深度学习面试100题](https://zhuanlan.zhihu.com/c_140166199)
[面试笔记](https://github.com/imhuay/Algorithm_Interview_Notes-Chinese)

## 复习

### [贝叶斯](https://zhuanlan.zhihu.com/p/30926070)

### 神经网络（MLP）的万能近似定理
1. 一个前馈神经网络如果具有至少一个非线性输出层，那么只要给予网络足够数量的隐藏单元，它就可以以任意的精度来近似任何从一个有限维空间到另一个有限维空间的函数。
2. 深度与宽度的关系: 一个单层的网络就足以表达任意函数，但是该层的维数可能非常大，且几乎没有泛化能力；此时，使用更深的模型能够减少所需的单元数，同时增强泛化能力（减少泛化误差）。参数数量相同的情况下，浅层网络比深层网络更容易过拟合。

### 连通域
1. 连通区域（Connected Component）一般是指图像中具有相同像素值且位置相邻的前景像素点组成的图像区域（Region，Blob）。通常连通区域分析处理的对象是一张二值化后的图像。
2. 方法：Two-Pass法，Seed-Filling种子填充法，见[CSDN](https://blog.csdn.net/liangchunjiang/article/details/79431339)

### K-means
1. 聚类的目的是找到每个$x_i \in x={x_1,...,x_m}$潜在的类别$y$
2. K-means算法是将样本聚类成$k$个簇（cluster）
3. 算法
  1. 随机选取$k$个聚类质心（cluster centroids）$u_1,...,u_k$
  2. 重复直至收敛
     1. 对每一个$x_i$计算其应所属的类c_i别$c_i=\arg\min_j||x_i-u_j||^2$
     2. 对每一个类，重新计算其质心$u_j=\frac{\sum_{i=1}^m1\{c_i=j\}x_i}{\sum_{i=1}^m1\{c_i=j\}}$
4. 收敛性
    1. 畸变函数$J(c,u)=\sum_{i=1}^m||x_i-u_{c_i}||$, 即每个样本点到其质心的距离平方和
    2. K-means是要将$J$调整到最小
    3. 可以固定每个类的质心$u$，调整每个样例的所属的类别$c$来让$J$函数减少
    4. 固定$c$，调整每个类的质心$u$也可以使$J$减小
    5. 当$J$递减到最小时，$u$和$c$也同时收敛。（在理论上，可以有多组不同的$u$和$c$值能够使得J取得最小值，但这种现象实际上很少见）

### 距离
1. 欧氏距离 $d=\sqrt{(\bf x-y)^T(x-y)}=\sqrt{(x1-x2)^2+(y1-y2)^2}$
2. 曼哈顿距离 $d=|x_1-x_2|+|y_1-y_2|$
3. 闵可夫斯基距离 $d=(\bf |x-y|^p)^{1/p}$, $p=1$退化为曼哈顿距离，$p=2$退化为欧氏距离
3. 切比雪夫距离 $d=\max(|x_1-x_2|,|y_1-y_2|)$
4. 余弦距离 $cos\theta=\frac{x_1x_2+y_1y_2}{\sqrt{x_1^2+y_1^2}\sqrt{x_2^2+y_2^2}}$
5. Jaccard相似度 $J(A,B)=\frac{A\cap B}{A\cup B}$
6. 相关系数 $\rho=\frac{Cov(X,Y)}{\sqrt{Var(X)}\sqrt{Var(Y)}}=\frac{E((X-EX)(Y-EY))}{\sqrt{Var(X)}\sqrt{Var(Y)}}$,$Cov$为协方差,$Var$为方差,$\rho=0$描述$X,Y$之间不存在**线性**关系，“不相关”是一个比“独立”要弱的概念
> 方差针对一维数据$Var(X)=\frac{1}{n-1}(X_i-\bar X)(X_i-\bar X)$
> 协方差针对二维数据$Cov(X,Y)=\frac{1}{n-1}(X_i-\bar X)(Y_i-\bar Y)$
> 协方差矩阵针对多维数据, 是一个对称的矩阵
$\sum(X,Y,Z)=\left(\begin{array}c
Cov(X,X) & Cov(X,Y) & Cov(X,Z) \\
Cov(Y,X) & Cov(Y,Y) & Cov(Y,Z) \\
Cov(Z,X) & Cov(Z,Y) & Cov(Z,Z)
\end{array}\right)$

8. 马氏距离 $d=\sqrt{(\bf x-y)^T\sum^{-1}(x-y)}$

### [概率分布的距离度量](https://zhuanlan.zhihu.com/p/27305237)
1. KL散度(不具备交换性) $D_{KL}(P||Q)=\sum_{i=1}P(i)\log\frac{P(i)}{Q(i)}$
2. JS距离 $D_{JS}(P||Q)=0.5D_{KL}(P||M)+0.5D_{KL}(Q||M),M=0.5(P+Q)$
3. [Wasserstein距离](https://blog.chaofan.io/archives/earth-movers-distance-%E6%8E%A8%E5%9C%9F%E6%9C%BA%E8%B7%9D%E7%A6%BB)，描述分布Q能够变换成分布P所需要的最小代价
### [GAN的稳定性](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650757216&idx=3&sn=6d448901ef2b8350e0c9a83cf63f3d97&chksm=871a9c1eb06d150836bb9e73385a7ef59eb3a0dc06c2af412cc0a1470608a377fc5a20061936&mpshare=1&scene=23&srcid=#rd)
1. 使用 GAN 的弊端
   1. 模式崩溃：生成单一模式的的数据
   2. 收敛性：无法从loss变化中判断收敛性
2. 解决方法
   1. Wasserstein GAN: Wasserstein距离，即使真实的和生成的样本的数据分布没有交集，Wasserstein距离也是连续的
   2. [LSGAN](https://wiseodd.github.io/techblog/2017/03/02/least-squares-gan/): 对远离决策边界的生成样本进行惩罚，本质上将生成的数据分布拉向实际的数据分布
   3. 两个时间尺度上的更新规则: 通过不同的学习率, 生成器使用较慢的更新规则, 判别器使用较快的更新规则
   4. 自注意力机制：提供全局信息（远距离依赖）

### [基本卷积模块](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247487358&idx=2&sn=34d3dd9968f933f75870c9e843d37dc0&chksm=f9a27df1ced5f4e72298ca776835a25e11f9fc41258be346ed33bb1a749e751a98be9890232b&mpshare=1&scene=23&srcid=#rd)
1. 瓶颈模块: 1×1 卷积来降低通道数
2. Inception
![inception](./.assets/inception.jpg)
3. ResNet & ResNeXt
![Res](./.assets/Res.jpg)
4. Dence
![Dence](./.assets/Dence.jpg)
5. Squeeze-and-Excitation
![SE](./.assets/SE.jpg)

### 降低过拟合风险的方法
1. 数据增强
2. 降低模型复杂度
3. 权值约束（添加正则化项）: L1,L2 正则化
4. Dropout, Bagging, BN
5. 提前终止
6. 参数绑定与参数共享

### 梯度爆炸、消失的解决办法
1. 爆炸：梯度截断（gradient clipping）——如果梯度超过某个阈值，就对其进行限制
   1. 截断value
   2. 截断norm
2. 良好的参数初始化策略也能缓解梯度爆炸问题（权重正则化）
2. 使用线性整流激活函数，如 ReLU 等
4. BN
5. 消失：门卷积、残差

### [BN](https://www.cnblogs.com/guoyaohua/p/8724433.html)
1. BN 是一种正则化方法（减少泛化误差），主要作用有：
   1. 加速网络的训练（缓解梯度消失，支持更大的学习率）
   2. 防止过拟合
   3. 降低了参数初始化的要求。
2. 基本原理
   1. 针对mini-batch，在网络的每一层输入之前增加归一化处理，使输入的均值为 0，标准差为 1。目的是将数据限制在统一的分布下。(白化)
  $$\hat x = \frac{x-mean}{var}$$
   2. 但同时 BN 也降低了模型的拟合能力，破坏了之前学到的特征分布；为了恢复数据的原始分布，BN 引入了一个重构变换来还原最优的输入数据分布，下面的$\gamma,\beta$就是$bn.weight,bn.bias$
   $$ y = \gamma\hat x +\beta $$
   3. 测试时使用全局mean和var，即bn.running_mean,bn.running_var

### [L1/L2 范数正则化](https://blog.csdn.net/jinping_shi/article/details/52433975)
1. 相同点
限制模型的学习能力，通过限制参数的规模，使模型偏好于权值较小的目标函数，防止过拟合。
2. 不同点
   1. L1 正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择；一定程度上防止过拟合
   2. L2 正则化主要用于防止模型过拟合
   3. L1 适用于特征之间有关联的情况；L2 适用于特征之间没有关联的情况
### Dropout & Bagging
1. 给定一个大小为 ${\displaystyle n}$ 的训练集 ${\displaystyle D}$，Bagging算法从中均匀、有放回地（即使用自助抽样法）选出 ${\displaystyle m}$ 个大小为 ${\displaystyle n'}$ 的子集 ${\displaystyle D_{i}}$，作为新的训练集。在这 ${\displaystyle m}$ 个训练集上使用分类、回归等算法，则可得到 ${\displaystyle m}$ 个模型，再通过取平均值、取多数票等方法，即可得到Bagging的结果。
> 提高其准确率、稳定性的同时，通过降低结果的方差，避免过拟合的发生。

2. Dropout 通过参数共享提供了一种廉价的 Bagging 集成近似—— Dropout 策略相当于集成了包括所有从基础网络除去部分单元后形成的子网络。
![drop](./.assets/drop.jpg)

### CNN
1. 基本特点：局部连接、稀疏交互、参数共享
2. 输出尺寸计算公式为
$$n_{out} = (n_{in}+2*n_{padding}-n_{filter})/n_{slide}+1$$
> 有分数时取floor

3. [卷积的实现](https://hal.inria.fr/inria-00112631/document): im2col转换为矩阵相乘
4. 运算过程
   1. 前向: $y=x*W$ * 为矩阵相乘，x，W已展开
   2. 反向: $\nabla x=\nabla y * W^T$
   3. 参数梯度: $\nabla W=x^T * \nabla y$
   4. 参数更新: $W=W-\nabla W$
5. 卷积核
   1. 构造：kernel $[k_h, k_w, n_{in}, n_{out}]$，bias $[n_{out}]$
   2. 数量 $n_{in}* n_{out}$
6. 转置卷积
7. 空洞卷积（扩张卷积、膨胀卷积）
7. 空间可分离卷积：3x3的核分离为一个3x1和一个1x3
![ssconv](./.assets/ssconv.jpg)
8. [深度可分离卷积](https://yinguobing.com/separable-convolution/): 降低参数量
   1. Depthwise Convolution: 每个channel单独用一个kernel做，输出与输入channel数相同
   2. Pointwise Convolution, 用1*1的kernel融合channel信息
   ![sconv](./.assets/sconv.jpg)
9. 门卷积 $Y=Conv_1(X)\odot Sigmoid(Conv_2(X))$
   + 可缓解梯度消失：因为公式中有一个卷积没有经过激活函数，所以对这部分求导是个常数，所以梯度消失的概率很小
10. 分组卷积：相当于多次卷积，每次负责一部分channel，有利于提高并行程度
   ![groupconv](./.assets/groupconv.jpg)
### RNN
1. 种类
   1. Elman network：每个时间步都有输出，且隐藏单元之间有循环连接
   $h^t=tanh(Wx^t+Uh^t_b+b), o=softmax(wh^t+b)$
   ![EN](./.assets/EN.jpg)
   2. Jordan network：每个时间步都有输出，但是隐藏单元之间没有循环连接，只有当前时刻的输出到下个时刻的隐藏单元之间有循环连接
   $h^t=tanh(Wx^t+Uo^t_b+b), o=softmax(wh^t+b)$
   ![JN](./.assets/JN.jpg)
   3. 隐藏单元之间有循环连接，但只有最后一个时间步有输出
   ![RNN3](./.assets/RNN3.jpg)
2. 为什么梯度消失、爆炸
   1. 最大步长为 T 的 RNN 展开后相当于一个共享参数的 T 层前馈网络
   2. 解决方法：梯度截断
   3. 残差结构、LSTM、GRU
3. RNN中可以使用ReLu，但最好使用单位矩阵来初始化权重矩阵
4. LSTM：增加了遗忘门 f、输入门 i、输出门 o，以及一个内部记忆状态c，一般由输入x和上一时刻隐状态h计算
![LSTM-RNN](./.assets/LSTM-RNN.jpg)
   1. $$
   \begin{array}{l}
    i_t=\sigma(W_i*[x,h_{t-1}]+b_i) \\
    f_t=\sigma(W_f*[x,h_{t-1}]+b_f) \\
    o_t=\sigma(W_o*[x,h_{t-1}]+b_o) \\
    \widetilde c_t=\tanh(W_c*[x,h_{t-1}]+b_c) \\
    c_t=(f_t\odot c_{t-1})+(i_t\odot\widetilde  c_t) \\
    h_t=o_t\odot\tanh(c_t),
    \end{array}$$
   1. 长期记忆 $f\to 1,i\to 0$
   2. 短期记忆 $f\to 0,i\to 1$
   3. 长短期记忆 $f\to 1,i\to 1$
   4. hard gate：门只取0,1
   5. 窥孔机制：c也参与门的计算
5. GRU:
   ![GRU](./.assets/GRU.jpg)
   1. LSTM 中的遗忘门和输入门的功能有一定的重合，于是将其合并为一个更新门$z$, 控制前一时刻的状态信息被融合到当前状态中的程度
   2. 并使用重置门$r$代替输出门, 用于控制忽略前一时刻的状态信息的程度

### [梯度下降与SGD](https://zhuanlan.zhihu.com/p/31229539)
1. 基本的梯度下降法每次使用所有训练样本的平均损失来更新参数；
   1. 因此，经典的梯度下降在每次对模型参数进行更新时，需要遍历所有数据；
   2. 当训练样本的数量很大时，这需要消耗相当大的计算资源，在实际应用中基本不可行。
2. 随机梯度下降（SGD）每次使用单个样本的损失来近似平均损失
3. 小批量 SGD 的更新过程
   1. 在训练集上抽取指定大小（batch_size）的一批数据 ${(x,y)}$
   2. 【前向传播】将这批数据送入网络，得到这批数据的预测值 $y_pred$
   3. 计算网络在这批数据上的损失，用于衡量 $y_pred$ 和 $y$ 之间的距离
   4. 【反向传播】计算损失相对于所有网络中可训练参数的梯度 $g$
   5. 将参数沿着负梯度的方向移动，即 $W=W-lr*g$
4. batch size的影响
   1. **较大的批能得到更精确的梯度估计**；
   2. **较小的批能带来更好的泛化误差**, 但需要**较小的学习率**以保持稳定性，这意味着**更长的训练时间**
   3. 内存消耗和批的大小成正比
   4. 通常使用 2 的幂数作为批量大小可以获得更少的运行时间。
4. SDG的问题
   1. 放弃了**梯度的准确性**，仅采用一部分样本来估计当前的梯度；因此 SGD 对梯度的估计常常出现偏差，造成目标函数收敛不稳定，甚至不收敛的情况
   2. 无论是经典的梯度下降还是随机梯度下降，都可能陷入局部极值点，SGD 还可能遇到“峡谷”和“鞍点”两种情况
      1. 峡谷类似一个带有坡度的狭长小道，左右两侧是“峭壁”；在峡谷中，准确的梯度方向应该沿着坡的方向向下，但粗糙的梯度估计使其稍有偏离就撞向两侧的峭壁，然后在两个峭壁间来回震荡。
      2. 鞍点的形状类似一个马鞍，一个方向两头翘，一个方向两头垂，而中间区域近似平地；一旦优化的过程中不慎落入鞍点，优化很可能就会停滞下来。
5. 避免病态问题
   1. 导致病态的原因是问题的条件数（condition number）非常大。条件数大意味着目标函数在有的地方（或有的方向）变化很快、有的地方很慢，比较不规律，从而很难用当前的局部信息（梯度）去比较准确地预测最优点所在的位置，只能一步步缓慢的逼近最优点，从而优化时需要更多的迭代次数。
   2. 办法：随机梯度下降（SGD）、批量随机梯度下降动态的学习率、带动量的 SGD
5. 带动量的 SGD：一方面是为了解决“峡谷”和“鞍点”问题；一方面也可以用于SGD 加速。原始 SGD 每次更新的步长只是梯度乘以学习率；现在，步长还取决于**历史梯度序列的大小和方向**；当许多连续的梯度指向相同的方向时，步长会被不断增大；
$v_t=\alpha v_{t-1}-\eta g , \theta_{t+1}=\theta_t+v_t$，在实践中，$\alpha$的一般取 0.5, 0.9, 0.99，分别对应最大 2 倍、10 倍、100 倍的步长$(v=\frac{-\eta g}{1-\alpha})$
   1. 如果把原始的 SGD 想象成一个纸团在重力作用向下滚动，由于质量小受到山壁弹力的干扰大，导致来回震荡；或者在鞍点处因为质量小速度很快减为 0，导致无法离开这块平地。
   2. 动量方法相当于把纸团换成了铁球；不容易受到外力的干扰，轨迹更加稳定；同时因为在鞍点处因为惯性的作用，更有可能离开平地。
   3. 动量方法以一种廉价的方式模拟了二阶梯度（牛顿法）
6. AdaGrad: 独立地适应模型的每个参数,具有较大偏导的参数相应有一个较大的学习率，而具有小偏导的参数则对应一个较小的学习率, 引入小正数$\delta$
$$ r\leftarrow r+g^2,\Delta\theta=\frac{-\eta}{\delta+\sqrt r}* g$$
7. RMSProp: 解决 AdaGrad 方法中学习率过度衰减的问题,使用**指数衰减平均**,加入了一个超参数$\rho$ 用于控制衰减速率
$$\begin{array}l
r\leftarrow \mathbb E[g^2]_ t= \rho\mathbb E[g^2]_ {t-1}+(1-\rho)g^2 \\
RMS[g]_ t=\sqrt{\mathbb E[g^2]_ t+\delta} \\
\Delta\theta=\frac{-\eta}{RMS[g]_ t}* g
\end{array}$$
![RMSProp](./.assets/RMSProp.jpg)
8. AdaDelta：AdaDelta 的前半部分与 RMSProp 完全一致 $\Delta\theta=\frac{RMS[\theta]_ {t-1}}{RMS[g]_ t}* g$, **不需要全局学习率**
9. Adam: 在 RMSProp 方法的基础上更进一步
   1. 除了加入**历史梯度平方的指数衰减平均**（r）外，还保留了**历史梯度的指数衰减平均**（s），相当于**动量**。
   2. Adam 行为就像一个带有摩擦力的小球，在误差面上倾向于平坦的极小值。
   ![adam](./.assets/adam.jpg)
   3. 注意到，$s,r$ 需要初始化为0；且 $ρ1,ρ2$ 推荐的初始值都很接近1（0.9 和 0.999）,这将导致在训练初期 s 和 r 都很小（偏向于 0），从而训练缓慢, 因此，Adam 通过修正偏差来抵消这个倾向。

### 交叉熵损失相对均方误差损失的优势
交叉熵损失的导数中含激活函数的导数$\sigma'$（[具体推导](https://blog.csdn.net/sinat_35512245/article/details/78627450)），如果$\sigma=Sigmoid$，$\sigma'$可能很小

### 范数
1. L0: 向量中非零元素的个数
2. L1: 向量中所有元素的绝对值之和
3. L2: 向量中所有元素平方和的开方
4. Lp：$|x|_ p=(\sum_i|x_i|^p)^{1/p}$
5. L$\infty$: 向量中最大元素的绝对值，也称最大范数
6. Frobenius 范数：相当于作用于矩阵的 L2 范数 $||A||_ F=\sqrt{\sum_{i,j}A_{i,j}^2}$
7. 范数的应用：正则化——权重衰减/参数范数惩罚
8. 权重衰减的目的: 限制模型的学习能力，通过限制参数$\theta$的规模（主要是权重$W$的规模，偏置$b$不参与惩罚），使模型偏好于权值较小的目标函数，防止过拟合。

### 高斯分布的好处
$$\begin{array}l
N(x;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(x-\mu)^2}{2\sigma^2}) \\
N(x;\mu,\Sigma)=\frac{1}{\sqrt{(2\pi)^n det(\Sigma) }}\exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))
\end{array}$$
1. 我们想要建模的很多分布的真实情况是比较接近正态分布的。中心极限定理（central limit theorem）说明很多独立随机变量的和近似服从正态分布。这意味着在实际中，很多复杂系统都可以被成功地建模成正态分布的噪声，即使系统可以被分解成一些更结构化的部分。
2. 第二，在具有相同方差的所有可能的概率分布中，正态分布在实数上具有最大的不确定性。因此，我们可以认为正态分布是对模型加入的先验知识量最少的分布。

### 平移等变、不变性
1. 不变性：（局部）平移不变性是一个很有用的性质，尤其是当我们关心某个特征是否出现而不关心它出现的具体位置时。卷积核池化具有这种性质，**池化操作有助于卷积网络的平移不变性**
2. 等变性：如果一个函数满足输入改变，输出也以同样的方式改变这一性质，我们就说它是等变 (equivariant) 的。于卷积来说，如果令 g 是输入的任意平移函数，那么卷积函数对于 g 具有等变性。

### 迁移学习相关概念
1. 迁移学习：迁移学习和领域自适应指的是利用一个任务（例如，分布 P1）中已经学到的内容去改善另一个任务（比如分布 P2）中的泛化情况。
2. one-shot learning：只有少量标注样本的迁移任务被称为 one-shot learning。在大数据上学习 general knowledge，然后在特定任务的小数据上有技巧的 fine tuning。
3. zero-shot learning：没有标注样本的迁移任务被称为 zero-shot learning。假设学习器已经学会了关于动物、腿和耳朵的概念。如果已知猫有四条腿和尖尖的耳朵，那么学习器可以在没有见过猫的情况下猜测该图像中的动物是猫。

### 图模型、结构化概率模型相关概念
1. 有向图模型：有向图模型（directed graphical model）是一种结构化概率模型，也被称为信念网络（belief network）或者贝叶斯网络（Bayesian network）
2. 无向图模型：无向图模型（undirected graphical Model），也被称为马尔可夫随机场（Markov random field, MRF）或者是马尔可夫网络（Markov network）
3. 图模型的优点：减少参数的规模、统计的高效性、减少运行时间
4. 图模型如何用于深度学习：限玻尔兹曼机（RBM）
