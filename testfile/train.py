import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from puzzleEnv import padEnv
from model import A3C
from puzzleUtil import Util

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock,device, optimizer=None):
    torch.manual_seed(args.seed + rank)
    nrow = args.row_size
    ncol = args.col_size
    ncolor = args.color_size
    env = padEnv(nrow,ncol,ncolor,args.thres,-1,args.noDup)
    util = Util(nrow,ncol,ncolor)
    env.seed(args.seed + rank)

    observation_space = (ncol-1)*nrow + (nrow-1)*ncol
    action_space = max(nrow,ncol)**2*ncolor
    model = A3C(observation_space, action_space).to(device)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
    

    model.train()
    #初始化
    env.reset()
    #参数
    limit = []
    #随机起手
    pos=np.random.randint(0,[nrow,ncol]).tolist()
    limit =util.getLimit(pos)
    while True:
        # 更新参数
        model.load_state_dict(shared_model.state_dict())
        s = env.board.board
         #平铺
        transS,_ = util.autoOptim(s)
        x = torch.unsqueeze(torch.FloatTensor(x).to(device), 0)
        a = model(x)
        a = choose_action_custom(transS,limit,model)
        prob = F.softmax(logit, dim=-1)
        log_prob = F.log_softmax(logit, dim=-1)
        entropy = -(log_prob * prob).sum(1, keepdim=True)
        entropies.append(entropy)
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
        s = s_















        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            episode_length += 1
            value, logit, (hx, cx) = model((state.unsqueeze(0),
                                            (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            state, reward, done, _ = env.step(action.numpy())
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * \
                values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
