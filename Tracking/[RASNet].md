# Learning Attentions: Residual Attentional Siamese Network for High Performance Online Visual Tracking
[paper](http://www.dcs.bbk.ac.uk/~sjmaybank/CVPR18RASTrackCameraV3.3.pdf)
[github](https://github.com/foolwood/RASNet)

![RAS](./.assets/RAS.jpg)
## Dual Attention
![dualatt](./.assets/dualatt.jpg)
1. a **general attention** superimposed by **a residual attention**
$$ \rho=\overline\rho + \widetilde\rho $$
2. The general part $\overline\rho$ encodes a generality learning from all training samples
3. residual part $\widetilde\rho$ describes the distinctiveness between the live tracking target and the learnt common model

## Channel Attention
1. The channel attention net is composed by a dimension reduction layer with reduction ratio $r$ (set to 4), a ReLU, and then a dimension increasing layer with a sigmoid activation.
2. input: a $d$ channel feature $Z=[z_1,z_2,...,z_d],z_i\in R^{W\times H}$
3. output: $\widetilde Z=[\widetilde z_1,...,\widetilde z_d, \widetilde z_i\in R^{W\times H}]$
$$ \widetilde z_i=\beta_i\cdot z_i $$

## Thoughts
文章在SiamFC的框架上加入了三个attention，分别处理general的定位，空间pixel-level的分类，和channel轴上的特征选择。这三种attention都是惯用的思路，能够建立起更鲁棒的特征。之前的实验中，空间attention和channel attention在single-stage detector中的表现并不明显，或许可以借鉴这种做法。

空间attention已经很类似于mask了，只不过这里的attention都没有引入额外的监督。
