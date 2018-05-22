## Markov Decision Processes

### Markov processes
1. The future is independent of the past given the present
> 未来只与当前有关

2. 转移概率(矩阵)

### Markov reward processes (MRP)
1. 在MP基础上增加了reward function和discount factor
2. value function: 就是 discount reward
3. Bellman Equation: 当前的reward+此后的折扣回报
    1. immediate reward $R_{t+1}$
    2. discounted value of successor state$\gamma v(S_{t+1})$
    3. 解法: Dynamic programming, Monte-Carlo evaluation, Temporal-Difference learning

### Markov Decision Process
1. 在MRP基础上增加 action, 转移概率、reward都与action有关
2. policy: state条件下的action概率分布
3. Value Function
     1. state-value function $v$
     2. action-value function $q$
4. Bellman Expectation Equation：immediate reward + discounted value of successor
     1. state-value function
     2. action-value function
5. optimal state-value function: 对所有policy，最大的$v$
optimal action-value function: 对所有policy，最大的$q$
> 表示这个MDP的最好的可能的性能

6. Optimal Policy
