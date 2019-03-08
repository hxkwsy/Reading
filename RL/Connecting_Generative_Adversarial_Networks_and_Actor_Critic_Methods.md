# Connecting Generative Adversarial Networks and Actor-Critic Methods
[arXiv](https://arxiv.org/abs/1610.01945)

## Introduction
1. 相似点
   1. 信息流从一个模型（选择动作或生成样本）到另一个模型（评估第一个模型的输出）
   2. 第二个模型从环境中学习（reward/真实样本分布），第一个模型基于第二个模型的error进行学习

## Algorithms
1. GAN
$$\min_G\max_D\mathbb E_{w\sim p_{data}}[\log D(w)]+\mathbb E_{z\sim \mathcal N(0,I)}[1-\log D(G(z))]$$

2. AC
   1. the actor being the policy and the critic being the value function

3. GANs as a kind of Actor-Critic
   可GAN看做是action不影响环境的AC，整个过程可看做是无状态的MDP
