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

class A3C(nn.Module):
    def __init__(self,N_STATES,N_ACTIONS):
        super(DQN, self).__init__()
        self.eval_cn1 = nn.Conv2d(6,6,3)
        self.eval_cn2 = nn.Conv2d(6,6,4)
        self.eval_cn3 = nn.Conv2d(6,6,5)
        self.eval_cn4 = nn.Conv2d(6,1,1)
        self.eval_val_fc1 = nn.Linear(210, 512)
        self.eval_val_fc2 = nn.Linear(512, 256)
        self.eval_val_fc3 = nn.Linear(256, N_ACTIONS)

        self.apply(weights_init)

        self.learn_step_counter = 0     # 用于 target 更新计时
        self.memory_counter = 0         # 记忆库记数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr=LR)    # torch 的优化器
        self.loss_func = nn.MSELoss().to(device)   # 误差公式

    def forward(self, x):
        x1 = F.elu(self.eval_cn1(x))
        x2 = F.elu(self.eval_cn2(x))
        x3 = F.elu(self.eval_cn3(x))
        x4 = F.elu(self.eval_cn4(x))
        x=torch.cat((x1.view(-1,6*4*4),x2.view(-1,6*3*3),x3.view(-1,6*2*2),x4.view(-1,36)),dim=1)
        x = F.relu(self.eval_val_fc1(x))
        x = F.relu(self.eval_val_fc2(x))
        actions_value = self.eval_val_fc3(x)
        return actions_value

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
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)    # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        
        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss






class SharedAdam(optim.Adam):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss