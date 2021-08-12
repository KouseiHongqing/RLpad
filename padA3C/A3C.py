'''
函数说明: 
Author: hongqing
Date: 2021-08-12 17:37:26
LastEditTime: 2021-08-12 18:02:46
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from padA3C.sharedAdam import SharedAdam
from padA3C.model import A3C
from env.puzzleEnv import padEnv
from env.puzzleUtil import Util
from tensorboardX import SummaryWriter
from padA3C.A3Cutil import v_wrap,push_and_pull


class Worker(mp.Process):
    def __init__(self, gnet, opt, name,gamma):
        super(Worker, self).__init__()
        self.gamma = gamma
        self.name = 'w%02i' % name
        self.gnet, self.opt = gnet, opt
        self.lnet = A3C(N_STATES, N_ACTIONS)           # local network
        self.util = Util(nRow,nCol,nColor)#定义util
        if self.name == 'w00':
            self.env = padEnv(fps=0,limitsteps=args.maxsteps)
        else:
            self.env = padEnv(fps=-1,limitsteps=args.maxsteps)
        
    def run(self):
        total_step = 1
        totalreward=0
        combo = 0
        maxcomboget = 0
        self.env.reset()
        pos= []
        limit =[]
        buffer_s, buffer_a, buffer_r = [], [], []
        totalreward = 0.
        if self.name == 'w00':
            self.env.unwrapped()
        writer = SummaryWriter(log_dir='./logs/process{}'.format(self.name), comment='train')
        while True:
            if self.name == 'w00':
                self.env.render()
            transS = torch.Tensor(self.env.getBoard()).unsqueeze(0)
            a = self.lnet.choose_action(transS,limit)
            s_, r, done, combo,pos,limit = self.env.step(pos,a,combo)
            transS_ = torch.Tensor(self.env.getBoard()).unsqueeze(0)
            maxcomboget = max(maxcomboget,combo)
            totalreward += r
            buffer_a.append(a)
            buffer_s.append(transS)
            buffer_r.append(r)
            if done:  # done and print information 
                epochloss = push_and_pull(self.opt, self.lnet, self.gnet, transS_, buffer_s, buffer_a, buffer_r, self.gamma)
                buffer_s, buffer_a, buffer_r,buffer_s_ = [], [], [],[]
                # with SummaryWriter(log_dir='./logs/process{}'.format(self.name), comment='train') as writer: 
                writer.add_scalar('step/loss',epochloss,total_step)
                writer.add_scalar('step/maxcombo',maxcomboget,total_step)
                writer.add_scalar('step/totalreward',totalreward,total_step)
                self.env.reset()
                totalreward=0
                combo = 0
                maxcomboget = 0
                pos= []
                limit =[]
            s = s_
            total_step += 1

def A3Cmethod(arg):
    global device,nCol,nRow,nColor,N_ACTIONS,N_STATES,args

    args=arg

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    nCol = args.col_size

    nRow = args.row_size

    nColor = args.color_size
    
    N_ACTIONS = (nCol-1)*nRow + (nRow-1)*nCol  # 能做的动作

    N_STATES = max(nRow,nCol)**2*nColor  # 能获取的环境信息数
    
    gnet = A3C(N_STATES, N_ACTIONS)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
    # parallel training
    workers = [Worker(gnet, opt, i,args.gamma) for i in range(args.num_processes)]
    [w.start() for w in workers]