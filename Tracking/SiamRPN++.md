# SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks
[arXiv](https://arxiv.org/abs/1812.11703)
[zhihu](https://zhuanlan.zhihu.com/p/56254712)
## Analysis on Siamese Networks for Tracking
1. 严格的平移不变性约束
$$ f(z,x[\Delta\tau_j])=f(z,x)[\Delta\tau_j] $$
2. 结构对称
$$ f(z,x')=f(x',z) $$
3. SiamNet不能用deep net的原因
   1. padding会破坏平移不变性
   2. RPN需要一个不对称的feature来分类和回归

## ResNet-driven Siamese Tracking
![SiamRPN++](./.assets/SiamRPN++.jpg)

## Layer-wise Aggregation
Conv3~5 分别预测一次加权相加

## Depthwise Cross Correlation
![DW_XCorr](./.assets/DW_XCorr.jpg)
