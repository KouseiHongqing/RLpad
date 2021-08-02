'''
函数说明: 
Author: hongqing
Date: 2021-07-29 17:38:30
LastEditTime: 2021-08-02 14:09:16
'''
'''
函数说明: 
Author: hongqing
Date: 2021-07-29 17:38:30
LastEditTime: 2021-07-29 18:05:09
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
BATCH_SIZE=32
GAMMA=0.9
EPSILON=0.9
MEMORY_CAPACITY = 4000
TARGET_REPLACE_ITER=1000
N_ACTIONS=3
class Net(nn.Module):
    def __init__(self,N_STATES):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 60)
        self.fc1.weight.data.random_(0, 10)   # initialization
        self.fc2 = nn.Linear(60, 120)
        self.fc2.weight.data.random_(0, 10)   # initialization
        self.out = nn.Linear(120, 2)
        self.out.weight.data.random_(0, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self,N_STATES):
        self.N_STATES=N_STATES
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_net, self.target_net = Net(N_STATES).to(self.device), Net(N_STATES).to(self.device)
        self.learn_step_counter = 0     # 用于 target 更新计时
        self.memory_counter = 0         # 记忆库记数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.02)    # torch 的优化器
        self.loss_func = nn.MSELoss().to(self.device)   # 误差公式
    
    def load(self,savefile,i_episode):
        self.eval_net.load_state_dict(torch.load(savefile+i_episode))
        self.target_net.load_state_dict(torch.load(savefile+i_episode))
        
    def save(self,savefile,i_episode):
        torch.save(self.eval_net.state_dict(), savefile+str(i_episode)+'.ckpt')

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # 这里只输入一个 sample
        if np.random.uniform() < EPSILON:   # 选最优动作
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]     # return the argmax
        else:   # 选随机动作
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新
        
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:,:self.N_STATES]).to(self.device)
        b_a = torch.LongTensor(b_memory[:,self.N_STATES:self.N_STATES+1].astype(int)).to(self.device)
        b_r = torch.FloatTensor(b_memory[:,self.N_STATES+1:self.N_STATES+2]).to(self.device)
        b_s_ = torch.FloatTensor(b_memory[:,-self.N_STATES:]).to(self.device)
        # b_s = torch.unsqueeze(b_s,0)
        # b_s_ = torch.unsqueeze(b_s_,0)
        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        b_a = torch.squeeze(b_a,0)
        q_eval = self.eval_net(b_s).gather(0, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)    # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        
        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss