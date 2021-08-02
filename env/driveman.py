
'''
函数说明: 
Author: hongqing
Date: 2021-07-29 16:11:43
LastEditTime: 2021-08-02 14:11:06
'''
from DDPG import DDPG
# from env import Myenv
from drivegame import Myenv
import numpy as np
from DQN import DQN
import torch
import random
MEMORY_CAPACITY = 4000

memlen = 1
s_dim = 10
a_dim = 2
a_bound = 2
dqn=DQN(memlen)
env = Myenv()
index = 0 
for i_episode in range(1,400000):
    s = env.game_restart(400)
    totalloss = 0
    while True:
        a = dqn.choose_action(np.array([s]))
        #debug
        if(a==0 and env.player.pos==0):
            a=1
        if(a==1 and env.player.pos==400):
            a=0
        aa = a
        if(aa==0):
            aa=-1
        
        # 选动作, 得到环境反馈
        s_, r,done = env.step(True,aa*1)
        # 快速收敛
        if(r==1):
            r=0
        # 存记忆
        dqn.store_transition(s, a, r, s_)
        if dqn.memory_counter > MEMORY_CAPACITY:
            totalloss += dqn.learn() # 记忆库满了就进行学习
        if done:    # 如果回合结束, 进入下回合
            break
        index +=1
    if(i_episode%100==0):
        print("episode:{},islearn:{},loss:{},long:{}".format(i_episode,dqn.memory_counter > MEMORY_CAPACITY,totalloss,index))
    s = s_
    index = 0 
    if (i_episode%1000==0):
        torch.save(dqn.eval_net.state_dict(), 'dqn_last.ckpt')

# ddpg = DDPG(a_dim, s_dim, a_bound)
# var = 3  # control exploration
# for i in range(30000):
#     playermid = env.player.pos + env.player.width/2
#     emid = np.array(env.record[-11:-1][::-1]) + env.gen.width/2
#     s = playermid-emid
#     ep_reward = 0
#     for j in range(MAX_EP_STEPS):

#         # Add exploration noise
#         a = ddpg.choose_action(s)
#         a = np.clip(np.random.normal(a, var), -4, 4)    # add randomness to action selection for exploration
#         s_, r, done = env.step(a)

#         ddpg.store_transition(s, a, r / 10, s_)

#         if ddpg.pointer > MEMORY_CAPACITY:
#             var *= .9995    # decay the action randomness
#             ddpg.learn()

#         s = s_
#         ep_reward += r
#         if j == MAX_EP_STEPS-1:
#             print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
#             break
