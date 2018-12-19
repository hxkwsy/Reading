# PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization
[arXiv](https://arxiv.org/abs/1505.07427)
[iccv](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.pdf)
[Pytorch](https://capsulesbot.com/blog/2018/08/24/apolloscape-posenet-pytorch.html)
[Project](http://mi.eng.cam.ac.uk/projects/relocalisation/)


## Introdcution
1. Contribution:
   1. leverage transfer learning from recognition to relocalization with very large scale classification datasets.
   2. use structure from motion to automatically generate training labels (camera poses) from a video of the scene.
2. several issues faced by typical SLAM pipelines
   1. the need to store densely spaced keyframes
   2. the need to maintain separate mechanisms for appearance-based localization and landmark-based pose estimation
   3. a need to establish frame-to-frame feature correspondence
1. Metric SLAM:  focusing on creating a sparse [13, 11] or dense [16, 7] map of the environment.
2. Appearance-based localization: provides this coarse estimate by classifying the scene among a limited number of discrete locations.
   1. Scalable appearance-based localizers [4]: uses SIFT features [15] in a bag of words approach to probabilistically recognize previously viewed scenery.
   2. SIFT-based: [27,14,9,3]

## Model for deep regression of camera pose
从单目图像$I$估计相机位姿(pose vector) $p=[x,q]$
> $x$为3D position, $q$为orientation quaternion四元素(这种表达比正交旋转矩阵简单 simpler than the orthonormalization required of rotation matrices), $p$直接定义在全局坐标系下

### Simultaneously learning location and orientation
1. 损失
$$ loss(I)=||\hat x-x||_ 2+\beta||\hat q-\frac{q}{||q||}||_ 2 $$
> $q$本应该用球形距离(spherical distance)描述，但当$q$和$\hat q$接近时，欧拉距离和球形距离相似(becomes insignificant)

2. full 6-DOF pose 训练比单独训position或orientation更好，因为只有其中一种信息的话，CNN不能确定相机的pose. 不同分支训position或orientation也不行(it too was less effective)

3. 最优的$\beta$由训练结束时的 ratio between expected error of position and orientation确定. indoor: 120~750 and outdoor: 250~2000

### Architecture
1. base: GoogLeNet
2. 修改:
   1. 替换softmax classifiers为affine regressors. 每一个FC层输出7维的pose vector(3-d position 4-d orientation)
   2. 在最后的回归插入另一个FC层
   3. 归一化quaternion orientation到单位长度
3. 训练时，先resize到256，再随机crop一个224的图像，减去均值作为输入
4. 测试时，有两种方式
   1. single center crop
   2. densely with 128 uniformly spaced crops, the averaging the resulting pose vectors

## Experiment
1.  The indoor dataset contains many ambiguous and textureless features which make relocalization without this depth modality extremely difficult.
> 室内数据集在缺乏深度信息的情况下更难估计

## Learned
利用CNN回归一个7位的pose向量

## Pytorch
1. Local Response Normalization
AlexNet提出了LRN层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。局部响应归一化原理是仿造生物学上活跃的神经元对相邻神经元的抑制现象（侧抑制）。详见[CSDN](https://blog.csdn.net/yangdashi888/article/details/77918311)
2. Model
```tex
Sequential(
  (0): InceptionBlock(
    (branch_x1): Sequential(
      (0): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
    )
    (branch_x3): Sequential(
      (0): Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
    )
    (branch_x5): Sequential(
      (0): Conv2d(192, 16, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (3): ReLU(inplace)
    )
    (branch_proj): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
      (2): ReLU(inplace)
    )
  )
  (1): InceptionBlock(
    (branch_x1): Sequential(
      (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
    )
    (branch_x3): Sequential(
      (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
    )
    (branch_x5): Sequential(
      (0): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(32, 96, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (3): ReLU(inplace)
    )
    (branch_proj): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
      (2): ReLU(inplace)
    )
    (pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  )
  (2): InceptionBlock(
    (branch_x1): Sequential(
      (0): Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
    )
    (branch_x3): Sequential(
      (0): Conv2d(480, 96, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(96, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
    )
    (branch_x5): Sequential(
      (0): Conv2d(480, 16, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(16, 48, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (3): ReLU(inplace)
    )
    (branch_proj): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(480, 64, kernel_size=(1, 1), stride=(1, 1))
      (2): ReLU(inplace)
    )
  )
  (3): InceptionBlock(
    (branch_x1): Sequential(
      (0): Conv2d(512, 160, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
    )
    (branch_x3): Sequential(
      (0): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(112, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
    )
    (branch_x5): Sequential(
      (0): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(24, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (3): ReLU(inplace)
    )
    (branch_proj): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
      (2): ReLU(inplace)
    )
  )
  (4): InceptionBlock(
    (branch_x1): Sequential(
      (0): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
    )
    (branch_x3): Sequential(
      (0): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
    )
    (branch_x5): Sequential(
      (0): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(24, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (3): ReLU(inplace)
    )
    (branch_proj): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
      (2): ReLU(inplace)
    )
  )
  (5): InceptionBlock(
    (branch_x1): Sequential(
      (0): Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
    )
    (branch_x3): Sequential(
      (0): Conv2d(512, 144, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(144, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
    )
    (branch_x5): Sequential(
      (0): Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (3): ReLU(inplace)
    )
    (branch_proj): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
      (2): ReLU(inplace)
    )
  )
  (6): InceptionBlock(
    (branch_x1): Sequential(
      (0): Conv2d(528, 256, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
    )
    (branch_x3): Sequential(
      (0): Conv2d(528, 160, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
    )
    (branch_x5): Sequential(
      (0): Conv2d(528, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(32, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (3): ReLU(inplace)
    )
    (branch_proj): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1))
      (2): ReLU(inplace)
    )
    (pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  )
  (7): InceptionBlock(
    (branch_x1): Sequential(
      (0): Conv2d(832, 256, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
    )
    (branch_x3): Sequential(
      (0): Conv2d(832, 160, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
    )
    (branch_x5): Sequential(
      (0): Conv2d(832, 32, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(32, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (3): ReLU(inplace)
    )
    (branch_proj): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1))
      (2): ReLU(inplace)
    )
  )
  (8): InceptionBlock(
    (branch_x1): Sequential(
      (0): Conv2d(832, 384, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
    )
    (branch_x3): Sequential(
      (0): Conv2d(832, 192, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
    )
    (branch_x5): Sequential(
      (0): Conv2d(832, 48, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace)
      (2): Conv2d(48, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (3): ReLU(inplace)
    )
    (branch_proj): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1))
      (2): ReLU(inplace)
    )
  )
  (9): RegressionHead(
    (dropout): Dropout(p=0.7)
    (projection): Sequential(
      (0): AvgPool2d(kernel_size=5, stride=3, padding=0)
      (1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
      (2): ReLU(inplace)
    )
    (cls_fc_pose): Sequential(
      (0): Linear(in_features=2048, out_features=1024, bias=True)
      (1): ReLU(inplace)
    )
    (cls_fc_xy): Linear(in_features=1024, out_features=3, bias=True)
    (cls_fc_wpqr): Linear(in_features=1024, out_features=4, bias=True)
  )
  (10): RegressionHead(
    (dropout): Dropout(p=0.7)
    (projection): Sequential(
      (0): AvgPool2d(kernel_size=5, stride=3, padding=0)
      (1): Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1))
      (2): ReLU(inplace)
    )
    (cls_fc_pose): Sequential(
      (0): Linear(in_features=2048, out_features=1024, bias=True)
      (1): ReLU(inplace)
    )
    (cls_fc_xy): Linear(in_features=1024, out_features=3, bias=True)
    (cls_fc_wpqr): Linear(in_features=1024, out_features=4, bias=True)
  )
  (11): RegressionHead(
    (dropout): Dropout(p=0.5)
    (projection): AvgPool2d(kernel_size=7, stride=1, padding=0)
    (cls_fc_pose): Sequential(
      (0): Linear(in_features=1024, out_features=2048, bias=True)
      (1): ReLU(inplace)
    )
    (cls_fc_xy): Linear(in_features=2048, out_features=3, bias=True)
    (cls_fc_wpqr): Linear(in_features=2048, out_features=4, bias=True)
  )
)

```
out size
```tex
output_bf: torch.Size([75, 192, 28, 28])
output_3a: torch.Size([75, 256, 28, 28])
output_3b: torch.Size([75, 480, 14, 14])
output_4a: torch.Size([75, 512, 14, 14])
output_4b: torch.Size([75, 512, 14, 14])
output_4c: torch.Size([75, 512, 14, 14])
output_4d: torch.Size([75, 528, 14, 14])
output_4e: torch.Size([75, 832, 7, 7])
output_5a: torch.Size([75, 832, 7, 7])
output_5b: torch.Size([75, 1024, 7, 7])
```
### PoseLSTM
```tex
(9): RegressionHead(
    (dropout): Dropout(p=0.7)
    (projection): Sequential(
      (0): AvgPool2d(kernel_size=5, stride=3, padding=0)
      (1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
      (2): ReLU(inplace)
    )
    (cls_fc_pose): Sequential(
      (0): Linear(in_features=2048, out_features=1024, bias=True)
      (1): ReLU(inplace)
    )
    (cls_fc_xy): Linear(in_features=1024, out_features=3, bias=True)
    (cls_fc_wpqr): Linear(in_features=1024, out_features=4, bias=True)
    (lstm_pose_lr): LSTM(32, 256, batch_first=True, bidirectional=True)
    (lstm_pose_ud): LSTM(32, 256, batch_first=True, bidirectional=True)
  )
  (10): RegressionHead(
    (dropout): Dropout(p=0.7)
    (projection): Sequential(
      (0): AvgPool2d(kernel_size=5, stride=3, padding=0)
      (1): Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1))
      (2): ReLU(inplace)
    )
    (cls_fc_pose): Sequential(
      (0): Linear(in_features=2048, out_features=1024, bias=True)
      (1): ReLU(inplace)
    )
    (cls_fc_xy): Linear(in_features=1024, out_features=3, bias=True)
    (cls_fc_wpqr): Linear(in_features=1024, out_features=4, bias=True)
    (lstm_pose_lr): LSTM(32, 256, batch_first=True, bidirectional=True)
    (lstm_pose_ud): LSTM(32, 256, batch_first=True, bidirectional=True)
  )
  (11): RegressionHead(
    (dropout): Dropout(p=0.5)
    (projection): AvgPool2d(kernel_size=7, stride=1, padding=0)
    (cls_fc_pose): Sequential(
      (0): Linear(in_features=1024, out_features=2048, bias=True)
      (1): ReLU(inplace)
    )
    (cls_fc_xy): Linear(in_features=1024, out_features=3, bias=True)
    (cls_fc_wpqr): Linear(in_features=1024, out_features=4, bias=True)
    (lstm_pose_lr): LSTM(64, 256, batch_first=True, bidirectional=True)
    (lstm_pose_ud): LSTM(32, 256, batch_first=True, bidirectional=True)
  )
```
