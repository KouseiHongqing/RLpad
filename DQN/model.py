import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
def weights_init(m):
    classname = m.__class__.__name__
    # if classname.find('Conv') != -1:
    #     weight_shape = list(m.weight.data.size())
    #     fan_in = np.prod(weight_shape[1:4])
    #     fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
    #     w_bound = np.sqrt(6. / (fan_in + fan_out))
    #     m.weight.data.uniform_(-w_bound, w_bound)
    #     m.bias.data.fill_(0)
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

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

        self.apply(weights_init)
        self.v = 1

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

class TinyNet(nn.Module):
    def __init__(self,N_STATES,N_ACTIONS,device='cpu'):
        super(TinyNet, self).__init__()
        self.device = device
        self.cn = nn.Conv2d(3,18,3)
        self.fc1 = nn.Linear(18,48*4)
        self.fc2 = nn.Linear(48*4,48)
        self.fc3 = nn.Linear(48,12)

        # self.apply(weights_init)

    def forward(self, x):
        x = F.relu(self.cn(x))
        x= x.view(-1,18)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions_value = self.fc3(x)
        return actions_value

    def choose_action(self, x,limit,eval=False):
        # 这里只输入一个 sample
        eps = 0.9
        if(eval):
            eps = 1
        if np.random.uniform() <= eps:   # 选最优动作
            actions_value = self.forward(x)[0].cpu().data.numpy()
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
                action = np.random.randint(0, 12)
        return action

class DQN(object):
    def __init__(self,args,device='cpu',model='tiny'):
        self.args=args
        self.device = device
        self.N_ACTIONS = (args.col_size-1)*args.row_size + (args.row_size-1)*args.col_size  # 能做的动作
        self.N_STATES = max(args.row_size,args.col_size)**2*args.color_size  # 能获取的环境信息数
        if(model=='tiny'):
            self.eval_net, self.target_net = TinyNet(self.N_STATES,self.N_ACTIONS).to(device), TinyNet(self.N_STATES,self.N_ACTIONS).to(device)
        elif(model=='normal'):
            self.eval_net, self.target_net = Net(self.N_STATES,self.N_ACTIONS).to(device), Net(self.N_STATES,self.N_ACTIONS).to(device)
        self.eval_net.share_memory()
        self.learn_step_counter = 0     # 用于 target 更新计时
        self.memory_counter = 0         # 记忆库记数
        self.memory = np.zeros((self.args.memory_capacity, self.N_STATES * 2 + 2))     # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr=args.lr)    # torch 的优化器
        self.loss_func = nn.MSELoss().to(device)   # 误差公式
        self.version = 0
        
    
    def load(self,savefile,i_episode):
        self.eval_net.load_state_dict(torch.load(savefile+i_episode))
        self.target_net.load_state_dict(torch.load(savefile+i_episode))
        
    def save(self,savefile,i_episode):
        torch.save(self.eval_net.state_dict(), './weights/'+savefile+str(i_episode)+'.ckpt')

    def choose_action(self, x,limit):
        # 这里只输入一个 sample
        if np.random.uniform() < self.args.dqn_epsilon:   # 选最优动作
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
                action = np.random.randint(0, self.N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        s = np.array(s).flatten()
        s_ = np.array(s_).flatten()
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % self.args.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def batch_store_transition(self,transition):
        self.memory = np.vstack((self.memory,transition))
        self.memory = self.memory[-self.args.memory_capacity:,:]
        self.memory_counter = self.memory.shape[0]

    def learn(self):
        # target net 参数更新
        if self.learn_step_counter % self.args.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(self.args.memory_capacity, self.args.batch_size)
        b_memory = self.memory[sample_index, :]
        siiz = self.args.color_size * max(self.args.row_size,self.args.col_size)**2
        b_s = torch.FloatTensor(b_memory[:, :siiz]).to(self.device)
        b_s = b_s.reshape(-1,self.args.color_size,max(self.args.row_size,self.args.col_size),max(self.args.row_size,self.args.col_size))
        b_a = torch.LongTensor(b_memory[:, siiz:siiz+1].astype(int)).to(self.device)
        b_r = torch.FloatTensor(b_memory[:, siiz+1:siiz+2]).to(self.device)
        b_s_ = torch.FloatTensor(b_memory[:, -siiz:]).to(self.device)
        b_s_ = b_s_.reshape(-1,self.args.color_size,max(self.args.row_size,self.args.col_size),max(self.args.row_size,self.args.col_size))

        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + self.args.dqn_gamma * q_next.max(1)[0].view(self.args.batch_size, 1)    # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
