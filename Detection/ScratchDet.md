# ScratchDet: Training Single-Shot Object Detectors from Scratch
[arXiv](https://arxiv.org/abs/1810.08425)
[git](https://github.com/KimSoybean/ScratchDet)

## Introduction
1. 预训练的局限
   1. 分类任务有不同程度的平移不变性，所以需要下采样，但是局部的特征对检测更关键
   2. 不方便替换backbone

## Method
1. BN in backbone and detection head
2. 基于VGG16、ResNet101的SSD300在Pascal VOC上表现差不多
   1. 原因是downsampling operation in the first convolution layer (i.e., conv1 x with stride 2) of ResNet-101.
   2. 这个操作严重影响了检测，特别是小目标
3. Root-ResNet：第一个7x7的卷积换成两个3x3
