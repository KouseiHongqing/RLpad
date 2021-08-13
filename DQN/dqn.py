'''
函数说明: 
Author: hongqing
Date: 2021-08-13 13:02:17
LastEditTime: 2021-08-13 16:08:27
'''
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
from env.puzzleEnv import padEnv
from env.puzzleBoard import Board
from env.puzzleUtil import Util
import multiprocessing as mp
import math
from DQN.model import DQN,TinyNet
# from env.puzzleEnv import padEnv
# from env.puzzleUtil import Util
from tensorboardX import SummaryWriter

ctx = mp.get_context("spawn")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Worker(ctx.Process):
    def __init__(self, sharedModel,pipe,name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.pipe = pipe
        self.sharedModel = sharedModel
        self.net = TinyNet(N_STATES, N_ACTIONS)           # local network
        #从主进程获取网络
        self.net.load_state_dict(sharedModel.state_dict())
        self.util = Util(nRow,nCol,nColor)#定义util
        self.env = padEnv(fps=-1,row=nRow,col=nCol,color=nColor,limitsteps=20,Full=True)

    def run(self,):
        total_step = 1
        totalreward=0
        combo = 0
        maxcomboget = 0
        #刷新版面
        self.env.reset()
        pos= []
        limit = []
        version = self.pipe.recv()
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
                self.env.reset()
                pos= []
                limit =[]
                #更新网络 顺便同步 因为走到这一步会卡主，等待主进程发送数据
                newversion = self.pipe.recv()
                if(version!=newversion):
                    version = newversion
                    self.net.load_state_dict(self.sharedModel.state_dict())
            total_step += 1


class Animater(ctx.Process):
    def __init__(self,sharedModel):
        super(Animater, self).__init__()
        self.sharedModel=sharedModel
        self.net = TinyNet(N_STATES, N_ACTIONS)           # local network
        #从主进程获取网络
        self.net.load_state_dict(sharedModel.state_dict())
        self.env = padEnv(fps=10,row=nRow,col=nCol,color=nColor,limitsteps=20,Full=True)

    def run(self,):
        totalreward=0
        combo = 0
        maxcomboget = 0
        #刷新版面
        self.env.reset()
        self.env.unwrapped()
        pos= []
        limit =[]
        while(True):
            #平铺
            self.env.render()
            transS = torch.Tensor(self.env.getBoard()).unsqueeze(0)
            a = self.net.choose_action(transS,limit)
            _, r, done, combo,pos,limit = self.env.step(pos,a,combo)
            maxcomboget = max(maxcomboget,combo)
            totalreward += r
            if done:    # 如果回合结束, 进入下回合
                #初始化
                totalreward=0
                combo = 0
                maxcomboget = 0
                self.env.reset()
                pos= []
                limit =[]
                self.net.load_state_dict(self.sharedModel.state_dict())

def DQNmethod(arg):
    print('start training...')
    global device,nCol,nRow,nColor,N_ACTIONS,N_STATES,args
    args=arg
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    nCol = args.col_size

    nRow = args.row_size

    nColor = args.color_size
    
    N_ACTIONS = (nCol-1)*nRow + (nRow-1)*nCol  # 能做的动作
    
    N_STATES = max(nRow,nCol)**2*nColor  # 能获取的环境信息数
    dqn = DQN(args,device=device,model='tiny') # 定义 DQN 系统
    #启动进程
    # child_process_list = []
    pipe_dict = dict((i, (pipe1, pipe2)) for i in range(args.num_processes) for pipe1, pipe2 in (ctx.Pipe(),))
    # for i in range(args.num_processes):
    #     print('start process:{}'.format(i))
    #     pro = ctx.Process(target=processEpoch, args=(dqn.eval_net,pipe_dict[i],i))
    #     child_process_list.append(pro)
    workers = [Worker(dqn.eval_net,pipe_dict[i][1],i) for i in range(args.num_processes)]
    if(args.fps>=0):
        #追加动画
        workers.append(Animater(dqn.eval_net,))
    # 发送第一波数据 启动进程探索 
    [pipe_dict[i][0].send(dqn.version) for i in range(args.num_processes)]
    [p.start() for p in workers]
    #开始训练
    nowepoch = 0
    epochloss=0
    calreward = 0
    calloss = 0
    calcombo =0
    processwriter = [SummaryWriter('./logs/process{}'.format(i+1), comment='tiny_result') for i in range(args.num_processes)]
    # args.max_episode_length
    for i_episode in range(1,1000000):
        for i in range(args.num_processes):
            receive = pipe_dict[i][0].recv()
            # transS, a, r, transS = receive[0]
            totalreward = receive[1]
            maxcomboget = receive[2]
            for transS, a, r, transS_ in receive[0]:
                dqn.store_transition(transS, a, r, transS_)
                if dqn.memory_counter >= args.memory_capacity:
                    epochloss = dqn.learn() # 记忆库满了就进行学习
            nowepoch+=1
            print('process:{},episode:{},loss:{},maxcombo:{},totalreward:{},train started:{}'.format(i,nowepoch,epochloss,maxcomboget,totalreward,dqn.memory_counter > args.memory_capacity))
            processwriter[i].add_scalar('step/loss',epochloss,math.ceil(nowepoch/args.num_processes))
            processwriter[i].add_scalar('step/maxcombo',maxcomboget,math.ceil(nowepoch/args.num_processes))
            processwriter[i].add_scalar('step/totalreward',totalreward,math.ceil(nowepoch/args.num_processes))
            # if(nowepoch%1000==0):
                # dqn.save('tinyconv',nowepoch)
            pipe_dict[i][0].send(dqn.version)

            

