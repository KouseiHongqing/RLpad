'''
函数说明: 
Author: hongqing
Date: 2021-07-23 13:21:53
LastEditTime: 2021-08-13 13:01:58
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
import multiprocessing as mp
import math
from model import DQN
# from env.puzzleEnv import padEnv
# from env.puzzleUtil import Util
from tensorboardX import SummaryWriter

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

class Worker(mp.Process):
    def __init__(self, sharedModel,pipe,name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.pipe = pipe
        self.sharedModel = sharedModel
        self.net = Net(N_STATES, N_ACTIONS)           # local network
        #从主进程获取网络
        self.net.load_state_dict(sharedModel.state_dict())
        self.util = Util(nRow,nCol,nColor)#定义util
        self.env = padEnv(fps=-1,limitsteps=20)

    def processEpoch(self,):
        total_step = 1
        totalreward=0
        combo = 0
        maxcomboget = 0
        #刷新版面
        self.env.reset()
        pos= []
        limit = []
        version = self.pipe.recv()
        writer = SummaryWriter(log_dir='./logs/process{}'.format(self.name), comment='train')
         #传输数据
        pipedata=[]
        while(True):
            #平铺
            transS = torch.Tensor(self.env.getBoard()).unsqueeze(0)
            a = self.net.choose_action(transS,limit)
            _, r, done, combo,pos,limit = self.env.step(pos,a,combo)
            transS_ = torch.Tensor(self.env.getBoard()).unsqueeze(0)

            maxcomboget = max(maxcomboget,combo)
            totalreward += r
            # 传输记忆
            pipedata.append([transS, a, r, transS_])

            if done:    # 如果回合结束, 进入下回合
                self.pipe.send((pipedata,totalreward,maxcomboget))
                pipedata=[]
                #初始化
                totalreward=0
                combo = 0
                maxcomboget = 0
                writer.add_scalar('step/loss',epochloss,total_step)
                writer.add_scalar('step/maxcombo',maxcomboget,total_step)
                writer.add_scalar('step/totalreward',totalreward,total_step)
                self.env.reset()
                pos= []
                limit =[]
                #更新网络 顺便同步 因为走到这一步会卡主，等待主进程发送数据
                newversion = pipr.recv()
                if(version!=newversion):
                    version = newversion
                    net.load_state_dict(sharedModel.state_dict())
            s = s_
            total_step += 1

def main():
    print('start training...')
    dqn = DQN(device=device,model='tiny') # 定义 DQN 系统
    #启动进程
    child_process_list = []
    pipe_dict = dict((i, (pipe1, pipe2)) for i in range(processes) for pipe1, pipe2 in (ctx.Pipe(),))
    for i in range(processes):
        print('start process:{}'.format(i))
        pro = ctx.Process(target=processEpoch, args=(dqn.eval_net,pipe_dict[i],i))
        child_process_list.append(pro)
    # 发送第一波数据 启动进程探索 
    [pipe_dict[i][0].send(dqn.version) for i in range(processes)]
    [p.start() for p in child_process_list]
    #开始训练
    nowepoch = 0
    epochloss=0
    calreward = 0
    calloss = 0
    calcombo =0
    processwriter = [SummaryWriter('./logs/process{}'.format(i+1), comment='tiny_result') for i in range(processes)]
    # args.max_episode_length
    for i_episode in range(1,1000000):
        for i in range(processes):
            receive = pipe_dict[i][0].recv()
            # transS, a, r, transS = receive[0]
            totalreward = receive[1]
            maxcomboget = receive[2]
            for transS, a, r, transS_ in receive[0]:
                dqn.store_transition(transS, a, r, transS_)
                if dqn.memory_counter >= MEMORY_CAPACITY:
                    epochloss = dqn.learn() # 记忆库满了就进行学习
            nowepoch+=1
            print('process:{},episode:{},loss:{},maxcombo:{},totalreward:{},train started:{}'.format(i,nowepoch,epochloss,maxcomboget,totalreward,dqn.memory_counter > MEMORY_CAPACITY))
            processwriter[i].add_scalar('step/loss',epochloss,math.ceil(nowepoch/processes))
            processwriter[i].add_scalar('step/maxcombo',maxcomboget,math.ceil(nowepoch/processes))
            processwriter[i].add_scalar('step/totalreward',totalreward,math.ceil(nowepoch/processes))
            if(nowepoch%1000==0):
                dqn.save('tinyconv',nowepoch)
            pipe_dict[i][0].send(dqn.version)

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
            

