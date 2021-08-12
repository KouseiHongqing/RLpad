'''
函数说明: 
Author: hongqing
Date: 2021-07-13 11:19:24
LastEditTime: 2021-08-11 16:10:03
'''
import argparse
import os
import torch
from puzzleEnv import padEnv

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
parser.add_argument('--animation', default=False,
                    help='control animation player.default False')
parser.add_argument('--fps', type=int,default=10,
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



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.environ['OMP_NUM_THREADS'] = '1'

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    env = padEnv(args.row_size,arg.col_size,args.color_size,args.thres,args.fps,args.noDup)
    
    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()       


    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()           