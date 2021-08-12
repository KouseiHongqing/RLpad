'''
函数说明: 
Author: hongqing
Date: 2021-08-12 17:45:10
LastEditTime: 2021-08-12 17:51:30
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

def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    
    nCol = args.col_size

    nRow = args.row_size

    nColor = args.color_size

    N_ACTIONS = (nCol-1)*nRow + (nRow-1)*nCol  # 能做的动作

    N_STATES = max(nRow,nCol)**2*nColor  # 能获取的环境信息数
    
    gnet  = A3C(N_STATES, N_ACTIONS)
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
    lnet = A3C(N_STATES, N_ACTIONS)           # local network
    env = padEnv(fps=50,limitsteps=args.maxsteps)   
    
    total_step = 1
    totalreward=0
    combo = 0
    maxcomboget = 0
    env.reset()
    pos= []
    limit =[]
    buffer_s, buffer_a, buffer_r = [], [], []
    totalreward = 0.
    env.unwrapped()
    writer = SummaryWriter(log_dir='./logs/process', comment='train')
    while True:
        env.render()
        transS = torch.Tensor(env.getBoard()).unsqueeze(0)
        a = lnet.choose_action(transS,limit)
        s_, r, done, combo,pos,limit = env.step(pos,a,combo)
        transS_ = torch.Tensor(env.getBoard()).unsqueeze(0)
        maxcomboget = max(maxcomboget,combo)
        totalreward += r
        buffer_a.append(a)
        buffer_s.append(transS)
        buffer_r.append(r)
        if done:  # done and print information 
            epochloss = push_and_pull(opt, lnet, gnet, transS_, buffer_s, buffer_a, buffer_r, args.gamma)
            buffer_s, buffer_a, buffer_r = [], [], []
            print(totalreward)
            # with SummaryWriter(log_dir='./logs/process{}'.format(self.name), comment='train') as writer: 
            writer.add_scalar('step/loss',epochloss,total_step)
            writer.add_scalar('step/maxcombo',maxcomboget,total_step)
            writer.add_scalar('step/totalreward',totalreward,total_step)
            env.reset()
            totalreward=0
            combo = 0
            maxcomboget = 0
            pos= []
            limit =[]
        s = s_
        total_step += 1