# Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image

## Introduction
1. 单目深度的优势：small, low-cost, energy efficient, and ubiquitous in consumer electronic products.
相比而言，其他都有劣势：
   1. 3D LiDARs： cost-prohibitive
   2. Structured-light-based depth sensors (e.g. Kinect): sunlight-sensitive and power-consuming
   3.  stereo cameras
       1. require a large baseline
       2. careful calibration for accurate triangulation
       3. demands large amount of computation and usually fails at featureless regions.

## Method
1. Depth Sampling: 从GT中筛选一些稀疏点
2. 有监督训练
