'''
函数说明: 
Author: hongqing
Date: 2021-08-11 16:05:12
LastEditTime: 2021-08-13 17:21:52
'''
from padA3C.sharedAdam import SharedAdam
from padA3C.A3C import A3Cmethod
from padA3C.A3Ctest import test
from DQN.dqn import DQNmethod
import argparse
import os


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
parser.add_argument('--row-size', type=int,default=5,
                    help='board row size')
parser.add_argument('--col-size', type=int,default=6,
                    help='board col size')
parser.add_argument('--color-size', type=int,default=6,
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
parser.add_argument('--memory-capacity', type=int,default=10000,
                    help='DQN MEMORY_CAPACITY ')
parser.add_argument('--target_replace_iter', type=int,default=1000,
                    help='DQN TARGET_REPLACE_ITER ')
parser.add_argument('--batch-size', type=int,default=64,
                    help='DQN BATCH_SIZE ')
parser.add_argument('--dqn-epsilon', type=int,default=0.9,
                help='DQN EPSILON')
parser.add_argument('--dqn-gamma', type=int,default=0.9,
                help='DQN GAMMA')
parser.add_argument('--is-play', default=True,
            help='CTW(changetheworld) or not')  

os.environ['OMP_NUM_THREADS'] = '1'

args = parser.parse_args()



if __name__ == "__main__":
    import shutil
    if('logs' in os.listdir('.')):
        shutil.rmtree('logs')
    # A3Cmethod(args)  
    DQNmethod(args) 
    