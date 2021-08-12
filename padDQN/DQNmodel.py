import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path

# 超参数
BATCH_SIZE = 32            
LR = 0.001 # learning rate
EPSILON = 0.9              # 最优选择动作百分比
GAMMA = 0.9                # 奖励递减参数
TARGET_REPLACE_ITER = 1000  # Q 现实网络的更新频率
MEMORY_CAPACITY = 10000     # 记忆库大小

class Net(nn.Module):
    def __init__(self,N_STATES,N_ACTIONS):
        super(Net, self).__init__()
        self.cn1 = nn.Conv2d(6,6,3)
        self.cn2 = nn.Conv2d(6,6,4)
        self.cn3 = nn.Conv2d(6,6,5)
        self.cn4 = nn.Conv2d(6,1,1)
        self.val_fc1 = nn.Linear(210, 512)
        self.val_fc2 = nn.Linear(512, 256)
        self.val_fc3 = nn.Linear(256, N_ACTIONS)

    def forward(self, x):
        x1 = F.relu(self.cn1(x))
        x2 = F.relu(self.cn2(x))
        x3 = F.relu(self.cn3(x))
        x4 = F.relu(self.cn4(x))
        x=torch.cat((x1.view(-1,6*4*4),x2.view(-1,6*3*3),x3.view(-1,6*2*2),x4.view(-1,36)),dim=1)
    
        x = F.relu(self.val_fc1(x))
        x = F.relu(self.val_fc2(x))
        actions_value = self.val_fc3(x)
        return actions_value

    def choose_action(self, x,limit):
        x = torch.unsqueeze(torch.FloatTensor(x).to(device), 0)
        # 这里只输入一个 sample
        actions_value = self.forward(x)[0].cpu().data.numpy()
        if(len(limit)>0):
            for index,__ in enumerate(actions_value):
                if(not index in limit):
                    actions_value[index] = -1e9
        #选一个最大的动作
        action = actions_value.argmax()
        return action

    def load(self,savefile,i_episode):
        self.eval_net.load_state_dict(torch.load(savefile+i_episode))
        self.target_net.load_state_dict(torch.load(savefile+i_episode))
        
    def save(self,savefile,i_episode):
        torch.save(self.eval_net.state_dict(), './weights/'+savefile+str(i_episode)+'.ckpt')

    def choose_action(self, x,limit):
        x = torch.unsqueeze(torch.FloatTensor(x).to(device), 0)
        # 这里只输入一个 sample
        if np.random.uniform() < EPSILON:   # 选最优动作
            actions_value = self.eval_net.forward(x)[0].cpu().data.numpy()
            if(len(limit)>0):
                for index,__ in enumerate(actions_value):
                    if(not index in limit):
                        actions_value[index] = -1e9
            #选一个最大的动作
            action = actions_value.argmax()
        else:   # 选随机动作
            if(len(limit)>0):
                action = np.random.choice(limit,1,False)[0]
            else:
                action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        s = np.array(s).flatten()
        s_ = np.array(s_).flatten()
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
        ##conv
        siiz = colorSize * max(nRow,nCol)**2
        b_s = torch.FloatTensor(b_memory[:, :siiz]).to(device)
        b_s = b_s.reshape(-1,colorSize,max(nRow,nCol),max(nRow,nCol))
        b_a = torch.LongTensor(b_memory[:, siiz:siiz+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, siiz+1:siiz+2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -siiz:]).to(device)
        b_s_ = b_s_.reshape(-1,colorSize,max(nRow,nCol),max(nRow,nCol))
        # b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)
        # b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).to(device)
        # b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).to(device)
        # b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)    # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        
        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.version+=1
        return loss

    def choose_action_custom(x,limit,eval_,device):
        x = torch.unsqueeze(torch.FloatTensor(x).to(device), 0)
        # 这里只输入一个 sample
        if np.random.uniform() < EPSILON:   # 选最优动作
            if(device=='cpu'):
                actions_value = eval_.forward(x)[0].data.numpy()
            else:
                actions_value = eval_.forward(x)[0].cpu().data.numpy()
            if(len(limit)>0):
                for index,__ in enumerate(actions_value):
                    if(not index in limit):
                        actions_value[index] = -1e9
            #选一个最大的动作
            action = actions_value.argmax()
        else:   # 选随机动作
            if(len(limit)>0):
                action = np.random.choice(limit,1,False)[0]
            else:
                action = np.random.randint(0, N_ACTIONS)
        return action