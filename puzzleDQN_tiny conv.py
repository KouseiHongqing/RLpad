'''
函数说明: 
Author: hongqing
Date: 2021-07-23 13:21:53
LastEditTime: 2021-07-29 13:51:36
'''
'''
函数说明: 
Author: hongqing
Date: 2021-07-13 15:40:23
LastEditTime: 2021-07-23 17:54:16
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
# import pygame
# from pygame.locals import *
import multiprocessing as mp
import datetime
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
ctx = mp.get_context("spawn")
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




nRow = 3
nCol = 3
colorSize=3
isPlay = False
animationOn = False
animationfps=5

#CPU数
processes = 4

N_ACTIONS = (nCol-1)*nRow + (nRow-1)*nCol  # 能做的动作

N_STATES = max(nRow,nCol)**2*colorSize  # 能获取的环境信息数

N_DEPTH = colorSize +1
#input (6+1) *
class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()
        # self.cn1 = nn.Conv2d(6,12,6,padding=3)
        # self.cn2 = nn.Conv2d(12,24,5,padding=2)
        # self.cn3 = nn.Conv2d(24,48,4,padding=1)
        # self.cn4 = nn.Conv2d(48,96,3,padding=1)
        # self.cn5 = nn.Conv2d(96,2,1)
        # self.val_fc1 = nn.Linear(2*max(nRow,nCol)**2, 256)
        # self.val_fc2 = nn.Linear(256, N_ACTIONS)
        self.cn = nn.Conv2d(3,2,3,padding=1)
        self.val_fc1 = nn.Linear(2*max(nRow,nCol)**2, 256)
        self.val_fc2 = nn.Linear(256, N_ACTIONS)

        
    def forward(self, x):
        # x = F.relu(self.cn1(x))
        # x = F.relu(self.cn2(x))
        # x = F.relu(self.cn3(x))
        # x = F.relu(self.cn4(x))
        # x = F.relu(self.cn5(x))
        # x = x.view(-1, 2*max(nRow,nCol)**2)
        # x = F.relu(self.val_fc1(x))
        # actions_value = self.val_fc2(x)
        x = F.relu(self.cn(x))
        x = x.view(-1, 2*max(nRow,nCol)**2)
        x = F.relu(self.val_fc1(x))
        actions_value = self.val_fc2(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().to(device), Net().to(device)
        self.learn_step_counter = 0     # 用于 target 更新计时
        self.memory_counter = 0         # 记忆库记数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # 初始化记忆库
        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(),momentum=0.9,alpha=0.99,eps=1e-5, lr=LR)    # torch 的优化器
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
def choose_action_custom(x,limit,eval_):
        x = torch.unsqueeze(torch.FloatTensor(x).to(device), 0)
        # 这里只输入一个 sample
        if np.random.uniform() < EPSILON:   # 选最优动作
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



def processEpoch(pipe):
    totalreward=0
    combo = 0
    maxcomboget = 0
    board = Board(rowSize=nRow,colSize=nCol,colorSize=colorSize,limitsteps=100) # 定义版面
    #刷新版面
    board.initBoardnoDup(True)
    
    util = Util(nRow,nCol,colorSize)#定义util
    limit = []
    #转珠限制
    pos=np.random.randint(0,[nRow,nCol]).tolist()
    if(isPlay):
        limit =util.getLimit(pos)
    #从主进程获取网络
    net = pipe.recv()
    eval_ = net
    #传输数据
    pipedata=[]
    while(True):
        s = board.board
        #平铺
        transS,_ = util.autoOptim(s)
        a = choose_action_custom(transS,limit,eval_)
        # 选动作, 得到环境反馈
        s_, r, done, combo,pos,limit = board.step(pos,a,combo)
        transS_,_ = util.autoOptim(s_)
        maxcomboget = max(maxcomboget,combo)
        # 传输记忆
        # dqn.store_transition(transS, a, r, transS_)
        totalreward += r

        pipedata.append([transS, a, r, transS_])
        
        if done:    # 如果回合结束, 进入下回合
            #发送数据
            pipe.send((pipedata,totalreward,maxcomboget))
            pipedata=[]
            #初始化
            totalreward=0
            combo = 0
            maxcomboget = 0
            board.initBoardnoDup(True)
            pos=np.random.randint(0,[nRow,nCol]).tolist()
            #更新网络 顺便同步 因为走到这一步会卡主，等待主进程发送数据
            net = pipe.recv()
            eval_ = net

        s = s_

        # print('记忆库已存储:{}/{}'.format(dqn.memory_counter,MEMORY_CAPACITY))
    print('episode:{},total loss:{},maxcombo:{},totalreward:{},train started:{}'.format(i_episode,totalloss/board.steps,maxcomboget,totalreward,dqn.memory_counter > MEMORY_CAPACITY))

#动画
# if(animationOn):
#     pygame.init()
#     screen = pygame.display.set_mode((135*nCol, 135*nRow))
#     pygame.display.set_caption("PadAutomation")
#     background = pygame.image.load(r'E:\RLpaz\data\normal.png').convert()
#     red = pygame.image.load(r'E:\RLpaz\data\red.png').convert()
#     green = pygame.image.load(r'E:\RLpaz\data\green.png').convert()
#     yellow = pygame.image.load(r'E:\RLpaz\data\yellow.png').convert()
#     dark = pygame.image.load(r'E:\RLpaz\data\dark.png').convert()
#     blue = pygame.image.load(r'E:\RLpaz\data\blue.png').convert()
#     pink = pygame.image.load(r'E:\RLpaz\data\pink.png').convert()
#     switch = {1: red,
#                     2: blue,
#                     3: green,
#                     4: yellow,
#                     5: dark,
#                     6: pink,
#                     }

def main():
    print('start training...')
    dqn = DQN() # 定义 DQN 系统
    #启动进程
    pipe_dict = dict((i, (pipe1, pipe2)) for i in range(processes) for pipe1, pipe2 in (ctx.Pipe(),))
    child_process_list = []
    for i in range(processes):
        print('start process:{}'.format(i))
        pro = ctx.Process(target=processEpoch, args=(pipe_dict[i][1],))
        child_process_list.append(pro)
    # 发送第一波数据 启动进程探索 
    [pipe_dict[i][0].send(dqn.eval_net) for i in range(processes)]
    [p.start() for p in child_process_list]

    # animation = padEnv() #定义动画
    # util = Util(nRow,nCol,colorSize)#定义util
    savefile = 'pazzuleparams_tiny'
    loadfile = 'pazzuleparams_tiny_last1.ckpt'
    if(os.path.isfile(loadfile)):
        print('loading weights....')
        dqn.target_net.load_state_dict(torch.load(loadfile))
        dqn.eval_net.load_state_dict(torch.load(loadfile))
    #启动动画
    #animation.gameStart(fps=0)
    #开始训练
    nowepoch = 0
    epochloss=0
    calreward = 0
    calloss = 0
    calcombo =0
    pltcombo = []
    pltreward = []
    time1=datetime.datetime.now()
    for i_episode in range(1,10000000):
        for i in range(processes):
            receive = pipe_dict[i][0].recv()
            # transS, a, r, transS = receive[0]
            totalreward = receive[1]
            maxcomboget = receive[2]
            for transS, a, r, transS_ in receive[0]:
                dqn.store_transition(transS, a, r, transS_)
                if dqn.memory_counter > MEMORY_CAPACITY:
                    epochloss = dqn.learn() # 记忆库满了就进行学习
            nowepoch+=1
            calreward += totalreward
            calloss += epochloss
            calcombo += maxcomboget
            print('process:{},episode:{},loss:{},maxcombo:{},totalreward:{},train started:{}'.format(i,nowepoch,epochloss,maxcomboget,totalreward,dqn.memory_counter > MEMORY_CAPACITY))
            if(nowepoch%1000==0):
                time2 = datetime.datetime.now()
                print('1000epochs执行所花费时间:{},平均得分:{},平均loss:{}'.format((time2-time1),calreward/1000,calloss/1000))
                time1 = time2
                pltreward.append(calreward/1000)
                pltcombo.append(calcombo/1000)
                calreward = 0
                calloss = 0
                calcombo=0

                if(nowepoch%10000==0):
                    plt.clf()
                    plt.figure(figsize=(8,6))
                    plt.title('训练效果')
                    plt.xlabel('训练次数/1000epoch')
                    plt.ylabel('得分')
                    a = np.arange(len(pltreward))
                    plt.plot(a, pltreward)
                    plt.plot(a, pltcombo)
                    plt.savefig('pic\\'+nowepoch)
        [pipe_dict[i][0].send(dqn.eval_net) for i in range(processes)]

if __name__ == '__main__':
    main()
            

    
