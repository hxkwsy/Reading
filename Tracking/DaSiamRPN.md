# Distractor-aware Siamese Networks for Visual Object Tracking
[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zheng_Zhu_Distractor-aware_Siamese_Networks_ECCV_2018_paper.pdf)
[git](https://github.com/foolwood/DaSiamRPN)
[zhihu](https://zhuanlan.zhihu.com/p/42546692)

## Introduction
1. Siamese的3个问题
   1. 只能区别前景和**非语义**的背景
   2. 不能在线更新模型
   3. 局部搜索策略，不能处理完全遮挡或出视野的情况
2. Siamese Net相关工作
   1. SINT [31]:
   2. SiamFC [2]
   3. RASNet [36]: Residual Attentional Network
   4. GOTURN [8]
   5. CFNet [33]: interprets the correlation filters as a differentiable layer
   6. FlowTrack [40]: 引入运动信息
   7. SiamRPN [16]
3. Long-term Tracking
结合short-term tracker 和 detector
   1. TLD [10]
   2. Ma et al. [20]: KCF+detector
   3. MUSTer [9]: KCF+SIFT
   4. Fan and Ling [6]

## Distractor-aware Siamese Networks
### Distractor-aware trarning
1. 不同类别的正样本能促进泛化能力
   1. VID：视频数量4,000，每帧标注
   2. Youtube-BB [27]：视频数量200,000，每30帧标注
   3. 利用ImageNet Detection [28] and COCO Detection [18]进行数据扩增（translation, resize, grayscale etc.）
2. 语义的负样本对能提升判别能力
   1. SiamFC和SiamRPN的判别力不强的原因有
     1. 缺乏语义负样本对，大多数背景仅仅是背景而不是object，因此他们可以被轻易分类
     2. 类内干扰项不平衡，这些都会是hard negative samples
3. 数据扩增
   1. translation
   2. scale variations
   3. illumination changes
   4. 引入运动模糊

### Distractor-aware Incremental Learning
1. SiamFC and SiamRPN用cosine窗来抑制干扰
2. 相似性度量 $f(z,x)=\phi(z)\star\phi(x)+b\cdot1$
3. 用NMS选择potential distractors
4. 用下式rerank proposals, 选择得分最大的
$$q=\arg\max_{p_k\in P}f(z,p_k)-\frac{\hat\alpha\sum_{n=1}^n\alpha_if(d_i,p_k)}{\sum_{n=1}^n\alpha_i}=\arg\max_{p_k\in P}(\phi(z)-\frac{\hat\alpha\sum_{n=1}^n\alpha_i\phi(d_i)}{\sum_{n=1}^n\alpha_i})\star\phi(p_k)$$
> $p$是proposal，$d$是干扰，$\hat\alpha$可以控制distractor learning的程度

### DaSiamRPN for Long-term Tracking
1. iterative local-to-global search strategy，丢掉目标以后逐渐扩大搜索区域
1. SiamRPN丢掉目标以后，score不会下降，DaSiamRPN的score可以意识到丢掉目标
