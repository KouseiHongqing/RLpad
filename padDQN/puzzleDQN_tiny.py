'''
函数说明: 
Author: hongqing
Date: 2021-07-23 13:21:53
LastEditTime: 2021-08-11 17:49:36
'''
'''
函数说明: 
Author: hongqing
Date: 2021-07-13 15:40:23
LastEditTime: 2021-07-23 17:54:16
'''
import configparser
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os.path
from puzzleEnv import padEnv
from puzzleBoard import Board
from puzzleUtil import Util
# import pygame
# from pygame.locals import *
import multiprocessing as mp
import datetime
import math
from torch.utils.tensorboard import SummaryWriter
ctx = mp.get_context("spawn")
config = configparser.ConfigParser()
config.read("hyperConfig.conf", encoding="utf-8")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 超参数
BATCH_SIZE = config.getint('HYPERPARA','BATCH_SIZE')
# LR = config.getfloat('HYPERPARA','LR')                  # learning rate
EPSILON = config.getfloat('HYPERPARA','EPSILON')              # 最优选择动作百分比
GAMMA = config.getfloat('HYPERPARA','GAMMA')                 # 奖励递减参数
TARGET_REPLACE_ITER = config.getfloat('HYPERPARA','TARGET_REPLACE_ITER')  # Q 现实网络的更新频率
MEMORY_CAPACITY = config.getint('HYPERPARA','MEMORY_CAPACITY')     # 记忆库大小



LR = 0.001
nRow = 5
nCol = 6
colorSize=6
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

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().to(device), Net().to(device)
        self.eval_net.share_memory()
        self.learn_step_counter = 0     # 用于 target 更新计时
        self.memory_counter = 0         # 记忆库记数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr=LR)    # torch 的优化器
        self.loss_func = nn.MSELoss().to(device)   # 误差公式

        self.version = 0
    
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



def processEpoch(sharedModel,update_data):
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
    net = Net()
    net.load_state_dict(sharedModel.state_dict())
    version = sharedModel.version
    while(True):
        s = board.board
        #平铺
        transS,_ = util.autoOptim(s)
        a = choose_action_custom(transS,limit,net,'cpu')
        # 选动作, 得到环境反馈
        s_, r, done, combo,pos,limit = board.step(pos,a,combo)
        transS_,_ = util.autoOptim(s_)
        maxcomboget = max(maxcomboget,combo)
        # 传输记忆
        # dqn.store_transition(transS, a, r, transS_)
        totalreward += r

        update_data.append([transS, a, r, transS_])
        
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
            net.load_state_dict(pipe.recv())
        s = s_

        # print('记忆库已存储:{}/{}'.format(dqn.memory_counter,MEMORY_CAPACITY))
    print('episode:{},total loss:{},maxcombo:{},totalreward:{},train started:{}'.format(i_episode,totalloss/board.steps,maxcomboget,totalreward,dqn.memory_counter > MEMORY_CAPACITY))

def main():
    print('start training...')
    dqn = DQN() # 定义 DQN 系统
    #启动进程
    child_process_list = []
    for i in range(processes):
        print('start process:{}'.format(i))
        pro = ctx.Process(target=processEpoch, args=(dqn.eval_net,))
        child_process_list.append(pro)
    # 发送第一波数据 启动进程探索 
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
    processwriter = [SummaryWriter('./logs/process{}'.format(i+1), comment='tiny_result') for i in range(processes)]
    
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
            processwriter[i].add_scalar('step/loss',epochloss,math.ceil(nowepoch/processes))
            processwriter[i].add_scalar('step/maxcombo',maxcomboget,math.ceil(nowepoch/processes))
            processwriter[i].add_scalar('step/totalreward',totalreward,math.ceil(nowepoch/processes))

            if(nowepoch%10==0):
                time2 = datetime.datetime.now()
                with SummaryWriter('./logs/train', comment='tiny_result') as writer:
                    writer.add_scalar('avg/score', calreward/1000,nowepoch/1000)
                    writer.add_scalar('avg/loss', calloss/1000,nowepoch/1000)
               
                calreward = 0
                calloss = 0
                calcombo=0

                if(nowepoch%1000==0):
                    # plt.clf()
                    # plt.figure(figsize=(8,6))
                    # plt.title('训练效果')
                    # plt.xlabel('训练次数/1000epoch')
                    # plt.ylabel('得分')
                    # a = np.arange(len(pltreward))
                    # plt.plot(a, pltreward)
                    # plt.plot(a, pltcombo)
                    # plt.savefig('pic\\'+nowepoch)
                    dqn.save('tinyconv',nowepoch)
            pipe_dict[i][0].send(dqn.eval_net.cpu().state_dict())

def main2():
    print('start training...')
    dqn = DQN() # 定义 DQN 系统
    #启动进程
    
    nowepoch = 0
    epochloss=0
    calreward = 0
    calloss = 0
    calcombo =0
    pltcombo = []
    pltreward = []
    totalreward=0
    maxcomboget=0
    combo=0
    board = Board(rowSize=nRow,colSize=nCol,colorSize=colorSize,limitsteps=100) # 定义版面
    #刷新版面
    board.initBoardnoDup(True)
    
    util = Util(nRow,nCol,colorSize)#定义util
    limit = []
    #转珠限制
    
    pos=np.random.randint(0,[nRow,nCol]).tolist()
    for i_episode in range(1,10000000):
        while(True):
            s = board.board
            #平铺
            transS,_ = util.autoOptim(s)
            a = choose_action_custom(transS,limit,dqn.eval_net)
            # 选动作, 得到环境反馈
            s_, r, done, combo,pos,limit = board.step(pos,a,combo)
            transS_,_ = util.autoOptim(s_)
            maxcomboget = max(maxcomboget,combo)
            # 传输记忆
            # dqn.store_transition(transS, a, r, transS_)
            totalreward += r
            dqn.store_transition(transS, a, r, transS_)
            if dqn.memory_counter > MEMORY_CAPACITY:
                epochloss = dqn.learn() # 记忆库满了就进行学习

            if done:    # 如果回合结束, 进入下回合
                
                print('totalreward is:{}'.format(totalreward))
                #初始化
                totalreward=0
                combo = 0
                maxcomboget = 0
                board.initBoardnoDup(True)
                pos=np.random.randint(0,[nRow,nCol]).tolist()
            s = s_
            # transS, a, r, transS = receive[0]

if __name__ == '__main__':
    main()
            

