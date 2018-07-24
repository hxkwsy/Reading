# Structure Inference Net: Object Detection Using Scene-Level Context and Instance-Level Relationships
[arXiv](https://arxiv.org/abs/1807.00119)

## Introduction
1. **Sequentially** object detection
regarded as not only a cognition problem, but also an inference problem which is based on contextual information with object fine-grained details. 

2. Graph problem
![graph](/assets/graph.jpg)
     1. objects are nodes of the graph 
     2. object relationships are edges of the graph
     3. object will receive messages from the scene and other objects that are highly correlated with it.

## Method
![SIN](/assets/SIN.jpg)
1. Graphical Modeling
    1. a graph $G=(V,E,s)$, $v\in V$ is region proposals, $e\in E$ is the edge (relationship) between each pair of object nodes
    2. RPN + NMS to choose a fixed number of ROIs. For each ROI $v_i$, extract the visual feature $f^v_i$ using ROI pooling and FC layer
    > 提取固定个数的ROI, 计算其特征作为图的node
    3. the whole image visual feature $f^s$ is extracted as the scene representation 
    > 全图特征作为 scene representation
    4. For directed edge $e_{j\to i}$ from $v_j$ to $v_i$, we use use both the spatial feature and visual feature to compute a scalar, which represents the influence of $v_j$ on $v_i$
    > 有向边是一个标量，由spatial feature(box位置) 和 visual feature 计算
2. Message Passing
For each node, encode the messages passed from the scene and other nodes to it
使用GRU来融合信息
![gru](/assets/gru.jpg)
    
    1. encode message from scene
        1. initial state: the fine-grained object details 
        2. input: message from scene
    2. encode message from other objects:
        1. initial state: object details
        2. input:  integrated message from other nodes 
3. Structure Inference
![SI](/assets/SI.jpg)
    1. scene GRUs
    2. edge GRUs
        $$
        \begin{array}l
        m_i^e=\max_{j\in V}pooling(e_{j\to i}*f_j^v) \\
        e_{j\to i}=relu(W_pR^p_{j\to i})* tanh(W_v[f_i^v,f_j^v]) \\
        R^p_{j\to i}=[w_i,h_i,s_i,w_j,h_j,s_j,\frac{(x_i-x_j)}{w_j},\frac{(y_i-y_j)}{h_j},\frac{(x_i-x_j)^2}{w_j^2},\frac{(y_i-y_j)^2}{h_j^2},\log(w_i/w_j),\log(h_i/h_j)]
        \end{array}
        $$
        > Using maxpooling can extract the most important message. $s$ is area
    3. $$ h_{t+1}=\frac{h_{t+1}^s+h_{t+1}^e}{2} $$
    > 两个GRU输出的平均