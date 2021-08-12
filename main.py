'''
函数说明: 
Author: hongqing
Date: 2021-08-11 16:05:12
LastEditTime: 2021-08-12 15:25:29
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from padA3C.sharedAdam import SharedAdam
from env.puzzleEnv import padEnv
from env.puzzleUtil import Util
from padA3C.model import A3C
import argparse
import os
import numpy as np
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae-lambda', type=float, default=1.00,
                    help='lambda parameter for GAE (default: 1.00)')
# parser.add_argument('--entropy-coef', type=float, default=0.01,
#                     help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('--fps', type=int,default=-1,
                    help='control animation fps.default 10')
parser.add_argument('--row_size', type=int,default=5,
                    help='board row size')
parser.add_argument('--col_size', type=int,default=6,
                    help='board col size')
parser.add_argument('--color_size', type=int,default=6,
                    help='board color size')
parser.add_argument('--thres', type=int,default=3,
                    help='combo thres')
parser.add_argument('--noDup', default=True,
                    help='duplicated')
parser.add_argument('--maxsteps', type=int,default=100,
                    help='MAX_EP')
parser.add_argument('--updateiter', type=int,default=10,
                    help='UPDATE_GLOBAL_ITER')
parser.add_argument('--method', type=str,default='A3C',
                    help='algorithm method')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
os.environ['OMP_NUM_THREADS'] = '1'

args = parser.parse_args()

torch.manual_seed(args.seed)

#env = padEnv()
nCol = args.col_size

nRow = args.row_size

nColor = args.color_size

N_ACTIONS = (nCol-1)*nRow + (nRow-1)*nCol  # 能做的动作

N_STATES = max(nRow,nCol)**2*nColor  # 能获取的环境信息数

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

def push_and_pull(opt, lnet, gnet, s_,bs, ba, br, gamma):

    v_s_ = lnet.forward(s_)[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))
    

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())

    return loss
    
def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )  

class Worker(mp.Process):
    def __init__(self, gnet, opt, name):
        super(Worker, self).__init__()
        
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
                epochloss = push_and_pull(self.opt, self.lnet, self.gnet, transS_, buffer_s, buffer_a, buffer_r, args.gamma)
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

def A3Cmethod():
    gnet = A3C(N_STATES, N_ACTIONS)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
    # parallel training
    workers = [Worker(gnet, opt, i ) for i in range(args.num_processes)]
    [w.start() for w in workers]

def test():
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
if __name__ == "__main__":
    if(args.method == 'A3C'):
        A3Cmethod()
    if(args.method == 'test'):
        test()