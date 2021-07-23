'''
函数说明: 
Author: hongqing
Date: 2021-07-22 17:22:17
LastEditTime: 2021-07-22 17:23:51
'''
# 导入环境和学习方法
from env import ArmEnv
from rl import DDPG

# 设置全局变量
MAX_EPISODES = 500
MAX_EP_STEPS = 200

# 设置环境
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# 设置学习方法 (这里使用 DDPG)
rl = DDPG(a_dim, s_dim, a_bound)

# 开始训练
for i in range(MAX_EPISODES):
    s = env.reset()                 # 初始化回合设置
    for j in range(MAX_EP_STEPS):
        env.render()                # 环境的渲染
        a = rl.choose_action(s)     # RL 选择动作
        s_, r, done = env.step(a)   # 在环境中施加动作

        # DDPG 这种强化学习需要存放记忆库
        rl.store_transition(s, a, r, s_)

        if rl.memory_full:
            rl.learn()              # 记忆库满了, 开始学习

        s = s_                      # 变为下一回合