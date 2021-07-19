'''
函数说明: 
Author: hongqing
Date: 2021-07-13 15:40:23
LastEditTime: 2021-07-19 10:13:33
'''
import configparser
from numpy.core.shape_base import hstack
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path
from puzzleMain import Board
from puzzleUtil import Util
import pygame
from pygame.locals import *
config = configparser.ConfigParser()
config.read("hyperConfig.conf", encoding="utf-8")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 超参数
BATCH_SIZE = config.getint('HYPERPARA','BATCH_SIZE')
LR = config.getfloat('HYPERPARA','LR')                  # learning rate
EPSILON = config.getfloat('HYPERPARA','EPSILON')              # 最优选择动作百分比
GAMMA = config.getfloat('HYPERPARA','GAMMA')                 # 奖励递减参数
TARGET_REPLACE_ITER = config.getfloat('HYPERPARA','TARGET_REPLACE_ITER')  # Q 现实网络的更新频率
MEMORY_CAPACITY = config.getint('HYPERPARA','MEMORY_CAPACITY')     # 记忆库大小

(6-1)*5 +(5-1)*6
N_ACTIONS = 49  # 能做的动作
#5*4*4
N_STATES = 30*6   # 能获取的环境信息数

nRow = 5
nCol = 6
colorSize=6

animationOn = False
animationfps=5

class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, N_STATES//colorSize)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out1 = nn.Linear(N_STATES//colorSize, N_STATES*3//colorSize)
        self.out1.weight.data.normal_(0, 0.1)   # initialization
        self.out2 = nn.Linear(N_STATES*3//colorSize, N_STATES*3)
        self.out2.weight.data.normal_(0, 0.1)   # initialization
        self.out3 = nn.Linear(N_STATES*3, N_STATES*3)
        self.out3.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(N_STATES*3, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out1(x)
        x = F.relu(x)
        x = self.out2(x)
        x = F.relu(x)
        x = self.out3(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().to(device), Net().to(device)
        self.learn_step_counter = 0     # 用于 target 更新计时
        self.memory_counter = 0         # 记忆库记数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # torch 的优化器
        self.loss_func = nn.MSELoss().to(device)   # 误差公式
    
    def load(self,savefile,i_episode):
        self.eval_net.load_state_dict(torch.load(savefile+i_episode))
        self.target_net.load_state_dict(torch.load(savefile+i_episode))
        
    def save(self,savefile,i_episode):
        torch.save(self.eval_net.state_dict(), savefile+str(i_episode)+'.ckpt')

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
#输入空间转换
# def transActionSpace(mat,pos,nrow=5,ncol=6):
#     row,col = pos+1
#     indexi=0
#     if(row==nrow):
#         row -=1
#     if(col==ncol):
#         col -=1
#     lu = mat[:row,:col]
#     lu = np.pad(lu,[(nrow-row-1,0),(ncol-col-1,0)])
#     ld = mat[row:,:col]
#     ld = np.pad(ld,[(0,row-1),(ncol-col-1,0)])
#     ru = mat[:row,col:]
#     ru = np.pad(ru,[(nrow-row-1,0),(0,col-1)])
#     rd = mat[row:,col:]
#     rd = np.pad(rd,[(0,row-1),(0,col-1)])
#     return lu,ld,ru,rd
if(animationOn):
    pygame.init()
    screen = pygame.display.set_mode((135*nCol, 135*nRow))
    pygame.display.set_caption("PadAutomation")
    background = pygame.image.load(r'E:\RLpaz\data\normal.png').convert()
    red = pygame.image.load(r'E:\RLpaz\data\red.png').convert()
    green = pygame.image.load(r'E:\RLpaz\data\green.png').convert()
    yellow = pygame.image.load(r'E:\RLpaz\data\yellow.png').convert()
    dark = pygame.image.load(r'E:\RLpaz\data\dark.png').convert()
    blue = pygame.image.load(r'E:\RLpaz\data\blue.png').convert()
    pink = pygame.image.load(r'E:\RLpaz\data\pink.png').convert()
    switch = {1: red,
                    2: blue,
                    3: green,
                    4: yellow,
                    5: dark,
                    6: pink,
                    }


dqn = DQN() # 定义 DQN 系统
board = Board() # 定义版面
# animation = padEnv() #定义动画
util = Util(nRow,nCol,colorSize)#定义util
savefile = 'pazzuleparams'
# if(os.path.isfile(savefile)):
#     dqn.load_state_dict(torch.load(savefile))
#启动动画
#animation.gameStart(fps=0)
#开始训练
for i_episode in range(1,10000000):
    totalloss = 0
    totalreward=0
    combo = 0
    maxcomboget = 0
    #刷新版面
    board.initBoardnoDup()
    pos=np.random.randint(0,[nRow,nCol]).tolist()
    #转珠限制
    limit =util.getLimit(pos)
    if(i_episode % 100000 ==0):
        dqn.save(savefile,i_episode)
        dqn.save(savefile,'_last')
    
    while(True):
        s = board.board
        if(animationOn and board.steps%5==0):
            screen.fill(0)
            for i in range(5):
                for j in range(6):
                    photo = switch[s[i][j]]
                    screen.blit(photo, (j*135,i*135))
            pygame.display.update()
        #平铺
        transS = s.reshape(1,-1)[0]
        transS = util.onehot(transS)
        a = dqn.choose_action(transS,limit)
        # 选动作, 得到环境反馈
        s_, r, done, combo,pos,limit = board.step(pos,a,combo)
        transS_ = util.boardTrans(s_.reshape(1,-1)[0])
        transS_ = util.onehot(transS_)
        maxcomboget = max(maxcomboget,combo)
        # 存记忆
        dqn.store_transition(transS, a, r, transS_)
        totalreward += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            totalloss += dqn.learn() # 记忆库满了就进行学习
            

        if done:    # 如果回合结束, 进入下回合
            break

        s = s_

        # print('记忆库已存储:{}/{}'.format(dqn.memory_counter,MEMORY_CAPACITY))
    print('episode:{},total loss:{},maxcombo:{},totalreward:{},train started:{}'.format(i_episode,totalloss/board.steps,maxcomboget,totalreward,dqn.memory_counter > MEMORY_CAPACITY))
    if(animationOn):
        for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()

