'''
函数说明: 
Author: hongqing
Date: 2021-08-11 15:37:42
LastEditTime: 2021-08-12 13:38:19
'''
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

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

class A3C(nn.Module):
    def __init__(self,N_STATES,N_ACTIONS):
        super(A3C, self).__init__()
        self.eval_cn1 = nn.Conv2d(6,6,3)
        self.eval_cn2 = nn.Conv2d(6,6,4)
        self.eval_cn3 = nn.Conv2d(6,6,5)
        self.eval_cn4 = nn.Conv2d(6,1,1)

        self.actor_linear1 = nn.Linear(210, 512)
        self.actor_linear2 = nn.Linear(512, N_ACTIONS)
        
        self.critic_linear1 = nn.Linear(210, 512)
        self.critic_linear2 = nn.Linear(512, 1)

        
        self.distribution = torch.distributions.Categorical

        self.apply(weights_init)

        self.actor_linear1.weight.data = normalized_columns_initializer(
            self.actor_linear1.weight.data, 0.01)
        self.actor_linear1.bias.data.fill_(0)
        self.actor_linear2.weight.data = normalized_columns_initializer(
            self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)
        self.critic_linear1.weight.data = normalized_columns_initializer(
            self.critic_linear1.weight.data, 1.0)
        self.critic_linear2.weight.data = normalized_columns_initializer(
            self.critic_linear2.weight.data, 1.0)
        

    def forward(self, x):
        x1 = F.elu(self.eval_cn1(x))
        x2 = F.elu(self.eval_cn2(x))
        x3 = F.elu(self.eval_cn3(x))
        x4 = F.elu(self.eval_cn4(x))
        x=torch.cat((x1.view(-1,6*4*4),x2.view(-1,6*3*3),x3.view(-1,6*2*2),x4.view(-1,36)),dim=1)
        l = torch.tanh(self.critic_linear1(x))
        v = torch.tanh(self.actor_linear1(x))
        return self.actor_linear2(v),self.critic_linear2(l)



    def choose_action(self,s,limit=[]):
        self.eval()
        # s = torch.unsqueeze(torch.FloatTensor(x).to(device), 0)
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        if(len(limit)>0):
                for index,__ in enumerate(prob[0]):
                    if(not index in limit):
                        prob[0][index] = 0
        m = self.distribution(prob)
        return m.sample().item()

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss






