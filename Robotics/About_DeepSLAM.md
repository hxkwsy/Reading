[深度学习和slam结合的论文1](https://www.zhihu.com/question/66006923/answer/238601811)
[深度学习和slam结合的论文2](https://www.zhihu.com/question/66006923/answer/513174641?utm_source=qq&utm_medium=social&utm_oi=638711330211762176)
[7sences data](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
```tex
J. Shotton, B. Glocker, C. Zach, S. Izadi, A. Criminisi, and A. Fitzgibbon. Scene coordinate regression forests for camera relocalization in rgb-d images. IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), 2013.
```
[Cambridge Landmarks ](http://mi.eng.cam.ac.uk/projects/relocalisation/#results)
```tex
A. Kendall, M. Grimes, and R. Cipolla. Posenet: A convolutional network for real-time 6-dof camera relocalization. In IEEE International Conference on Computer Vision (ICCV), 2015.
```

[单目深度](https://zhuanlan.zhihu.com/p/47290598)
[CRF的通俗理解](https://www.jianshu.com/p/55755fc649b1)
[相机与图像](https://zhuanlan.zhihu.com/p/33583981)

[SLAM基础](https://zhuanlan.zhihu.com/p/23247395)
1. VO
   1. 特征匹配法(PnP, Perspective-N-Point): 特征点的移动来估计相机的移动
   2. 直接法：直接把图像中所有像素写进一个位姿估计方程，求出帧间相对运动。
2. 后端：处理VO的漂移
   1. 滤波
3. 建图
4. 回环检测：机器人识别曾到达场景的能力
   1. 词袋模型(Bag-of-Words, BoW): 图像中的视觉特征（SIFT, SURF等）聚类，然后建立词典，进而寻找每个图中含有哪些“单词”（word）
   2. 看做分类问题
