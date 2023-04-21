import numpy as np
import os
import torch
from NEWenvironment import Env
from utils import seed_torch
from ppo_agent import PPOAgent
import time
from env_setting import Setting
import json
from model_test import model_test
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import random
import sys
import plotstyle as plts

def clamp_number(num, a, b):
    return max(min(num, max(a, b)), min(a, b))

def seek_state(state_c,i):
    if i == args.user_num-1:
        remainder = 0
    else:
        remainder = i+1
    idx_other = [True if idx % args.user_num != remainder else False for idx in range(args.state_dim_c)]
    idx_other[0] = True
    state_i = state_c[0,idx_other]
    state_i = state_i.unsqueeze(dim=0)
    return state_i

def main(args, env_args):

    writer = SummaryWriter(comment='scalar')
    seed_torch(args.seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    if args.use_cuda:
        torch.cuda.set_device(args.device_num)
    device = torch.device('cuda' if args.use_cuda else 'cpu')
    # -------------get environment information------------

    ppo_agent = []
    for i in range(args.user_num):
        ppo_agent.append(
            PPOAgent(args,device))

    done_time = 0
    epis = 0

    user_num = args.user_num
    env = Env(args,device,env_args)

    action = torch.zeros(user_num)
    value = torch.zeros(user_num)
    action_log_probs = torch.zeros(user_num)
    file_path = os.path.join(args.root_path, 'file')
    result_path = file_path + '/result.npz'
    model_path = os.path.join(args.root_path, 'model')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    rewards = []
    server_rewards = []
    user_rewards = [[] for _ in range(user_num)]
    completion_ratio = [[] for _ in range(env.task_num)]

    ext_rewards = []
    int_rewards = []
    server_path = os.path.join(model_path, 'server_model.pt')
    exp_num = 0

    if args.load_model:
        # env.local_ppo_model.load_state_dict(torch.load(server_path))
        # for i, ppo in enumerate(ppo_agent):
        #     ppo_model_path = os.path.join(model_path, 'ppo_model' + str(i) + '.pt')
        #     ppo.local_ppo_model.load_state_dict(torch.load(ppo_model_path))
        model_test(args, env_args,writer)
        sys.exit()
    action_explo_c = np.zeros((user_num, args.max_episode_length))
    action_explo_s = np.zeros(args.max_episode_length)
    utility_explo_c = torch.zeros(user_num, args.max_episode_length)
    utility_explo_s = torch.zeros(args.max_episode_length)
    loss_clnt = torch.zeros(user_num,args.max_episode_length)
    loss_actor_clnt = torch.zeros(user_num,args.max_episode_length)
    loss_value_clnt = torch.zeros(user_num,args.max_episode_length)
    loss_server = torch.zeros(args.max_episode_length)
    loss_actor_serv = torch.zeros(args.max_episode_length)
    loss_value_serv = torch.zeros(args.max_episode_length)
    for epis in range(args.max_episode_length):
        print('---------------in episode ', epis, '-----------------------')
        av_value_loss = 0
        av_policy_loss = 0
        # 将服务端状态复位，用最新的模型重新采集数据
        state_s = env.reset('s')
        env.after_update(state_s)

        for explo in range(args.D):
            print('---------------in explorations', explo, '-----------------------')
            # 将客户端状态复位，用最新的模型重新采集数据
            state_c = env.reset('c')
            for i, agent in enumerate(ppo_agent):
                state_input = seek_state(state_c, i)
                agent.after_update(state_input)

            av_reward = torch.zeros(user_num)
            av_action = torch.zeros(user_num)
            interact_time = 0
            sum_user_reward = np.zeros(user_num)
            interact_time += 1
            # ----------------sample actions(no grad)------------------------
            with torch.no_grad():
                server_value,server_action,server_action_logprobs = env.act(state_s)
                print('server bonus:%f' % (env.R))
                action_explo_s[epis] = server_action
                state_s,state_c = env.step_stage1(state_s,state_c)
                for game in range(args.D):
                    for i, agent in enumerate(ppo_agent):
                        state_input = seek_state(state_c,i)
                        value[i], action[i], action_log_probs[i] = agent.act(state_input)
                        action_explo_c[i,epis] = action[i]
                    state_s,state_c, reward = env.step_stage2(action,state_s,state_c)
                    for i,agent in enumerate(ppo_agent):
                        state_next = seek_state(state_c,i)
                        agent.insert(state_next.detach(), action[i].detach(), action_log_probs[i].detach(),
                                     value[i].detach(),
                                     reward[i].detach())
            exp_num +=1
            utility_explo_s[epis] = env.server_reward_cur
            if exp_num % args.action_std_decay_freq==0:
                for i, agent in enumerate(ppo_agent):
                    agent.decay_action_std(args.action_std_decay_rate, args.min_action_std)
                env.decay_action_std(args.action_std_decay_rate, args.min_action_std)

            for i, agent in enumerate(ppo_agent):
                utility_explo_c[i, epis] = reward[i]
                value_loss, policy_loss,ent_loss, total_loss = agent.update()
                loss_actor_clnt[i,epis] = policy_loss
                loss_value_clnt[i,epis] = value_loss
                loss_clnt[i,epis] = total_loss
            env.insert(state_s.detach(), server_action.detach(), server_action_logprobs.detach(), server_value.detach(),
                       torch.tensor(env.server_reward_cur))


            avg_reward = reward.numpy().mean()
            sum_user_reward += reward.numpy()

            av_reward += reward
            av_action += 0.2 * action.float()

            server_rewards.append(env.server_reward)

        av_value_loss /= user_num
        av_policy_loss /= user_num

        value_loss_s, policy_loss_s, ent_loss, total_loss_s = env.update()
        loss_actor_serv[epis] = policy_loss_s
        loss_value_serv[epis] = value_loss_s
        loss_server[epis] = total_loss_s
        for i in range(args.user_num):
            # writer.add_scalars('action/user_'+str(i),{'real':action_explo_c[i,epis],'nash':env.nash_useraction[i]},epis)
            # writer.add_scalars('utility/user_'+str(i),{'real':utility_explo_c[i,epis],'nash':env.nash_userreward[i]},epis)
            print('user %s|| real action %s' % (i,action_explo_c[i,epis]))
            print('user %s|| real utility %s' % (i, utility_explo_c[i,epis]))
        # writer.add_scalars('action/server',{'real':action_explo_s[explo],'nash':env.nash_action},epis)
        # writer.add_scalars('action/client_avg', {'real':action_explo_c.mean(),'nash':env.nash_useraction.mean()}, epis)
        # writer.add_scalars('utility/server',{'real':env.server_reward_cur,'nash':env.nash_reward},epis)
        # writer.add_scalars('utility/users_avg',{'real':avg_reward,'nash':env.nash_userreward.mean()},epis)
        print('server || action %s'%(action_explo_s[epis]))
        print('server || utility %s'%(env.server_reward_cur))


        for i in range(user_num):
            user_rewards[i].append(sum_user_reward[i] / args.exploration_steps)

        # writer.add_scalars('Loss/value_loss', {'server':value_loss_s,'client':av_value_loss}, epis)
        # writer.add_scalars('Loss/policy_loss', {'server':policy_loss_s,'client':av_policy_loss}, epis)
        # writer.add_scalars('Loss/ent_loss', {'server':ent_loss_s,'client':av_ent_loss}, epis)
        # writer.add_scalars('Loss/total_loss', {'server': total_loss_s, 'client': total_loss}, epis)
        # av_reward /= args.exploration_steps
        # av_action /= args.exploration_steps

    pltool = plts.plotstyle()
    pltool.plotscar(path=os.path.join('log','strategy_s.png'),x=range(args.max_episode_length),y=[action_explo_s],
                    xlabel='Episode',ylabel='Strategies of Server',legend=['server'])
    pltool.plotscar(path=os.path.join('log', 'utility_s.png'), x=range(args.max_episode_length), y=[utility_explo_s],
                    xlabel='Episode', ylabel='Utility of Server', legend=['server'])
    pltool.plotscar(path=os.path.join('log','strategy_c.png'),x=range(args.max_episode_length),y=[action_explo_c[0],action_explo_c[1]],
                    xlabel='Episode',ylabel='Strategies of Clients',legend=['client1','client2'])
    pltool.plotscar(path=os.path.join('log', 'utility_c.png'), x=range(args.max_episode_length),
                    y=[utility_explo_c[0], utility_explo_c[1]],
                    xlabel='Episode', ylabel='Utility of Clients', legend=['client1', 'client2'])
    pltool.plotscar(path=os.path.join('log','loss_s.png'),x=range(args.max_episode_length),y=[loss_server,loss_value_serv,loss_actor_serv],
                    xlabel='Episode',ylabel='Loss',legend=['total_loss','value_loss','actor_loss'])
    for i in range(args.user_num):
        pltool.plotscar(path=os.path.join('log','loss_clnt'+str(i)+'.png'),x=range(args.max_episode_length),y=[loss_clnt[i],loss_value_clnt[i],loss_actor_clnt[i]],
                        xlabel='Episode',ylabel='Loss',legend=['total_loss','value_loss','actor_loss'])
    np.savez(result_path, np.asarray(rewards), np.asarray(server_rewards), np.asarray(ext_rewards),
             np.asarray(int_rewards))
    # reward_profile.close()
    torch.save(env.local_ppo_model.state_dict(), server_path)
    for i, ppo in enumerate(ppo_agent):
        ppo_model_path = os.path.join(model_path, 'ppo_model' + str(i) + '.pt')
        torch.save(ppo.local_ppo_model.state_dict(), ppo_model_path)
    # model_test(args, env_args,writer)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    setup_seed(42)
    parser = argparse.ArgumentParser(description='Hyper-parameter setting for DRL-SIM.')
    # ------------------------------------- parameters that must be configured ---------------------------------
    parser.add_argument('--root_path', type=str, default='log', help='the path to save your results and models')
    parser.add_argument('--user_num', type=int, default=10, help='use cuda device to train models or not')

    # ------------------------------------- parameters that can be changed according to your need --------------
    parser.add_argument('--use-cuda', type=bool, default=False, help='use cuda device to train models or not')
    parser.add_argument('--device-num', type=int, default=0, help='cuda device number for training')
    parser.add_argument('--test-device-num', type=int, default=-1, help='cuda device number for testing')
    parser.add_argument('--max-episode-length', type=int, default=200)
    parser.add_argument('--exploration-steps', type=int, default=100)
    parser.add_argument('--max-test-length', type=int, default=100)
    parser.add_argument('--mini-batch-num', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ppo-epoch', type=int, default=4)
    parser.add_argument('--load_model', type=bool, default=False)
    # ------------------------------------- parameters that never recommend to be changed ---------------------
    parser.add_argument('--lr_actor', type=float, default=0.0003, help='optimizer learning rate')
    parser.add_argument('--lr_critic', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--ent-coeff', type=float, default=0.01)
    parser.add_argument('--value-coeff', type=float, default=0.1)
    parser.add_argument('--clip-coeff', type=float, default=1.0)
    parser.add_argument('--use-gae', type=bool, default=False)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--gae_param', type=float, default=0.95)
    parser.add_argument('--remain_budget', type=int, default=1e6)
    parser.add_argument('--remain_data', type=int, default=1000)
    parser.add_argument('--action_dim_c',type=int,default=1)
    parser.add_argument('--action_dim_s', type=int, default=1)
    parser.add_argument('--server_bias', type=float, default=80)
    parser.add_argument('--client_bias', type=float, default=5)

    parser.add_argument('--D', type=int, default=20,help='every D actions update model')
    parser.add_argument('--L', type=int, default=5,help='the window size of the state')
    parser.add_argument('--action_std_init', type=float, default=0.4)
    parser.add_argument('--action_std_decay_freq', type=int, default=400)
    parser.add_argument('--action_std_decay_rate', type=float, default=0.05)
    parser.add_argument('--min_action_std', type=float, default=0.1)
    args = parser.parse_args()
    args.state_dim_s = (args.user_num+1)*args.L
    args.state_dim_c = (args.user_num)*args.L+1
    # args.state_dim_s = 5
    # args.state_dim_c = args.user_num*2+1
    local_time = str(time.strftime("%Y/%m-%d/%H-%M-%S", time.localtime()))
    # local_time = '2022/10-30/21-06-53'
    Path(args.root_path).mkdir(parents=True,exist_ok=True)
    args.root_path = os.path.join(args.root_path, local_time)
    file_path = os.path.join(args.root_path, 'file')
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    with open(os.path.join(file_path, 'agent_args.txt'), 'a') as f:
        f.write(json.dumps(args.__dict__))

    env_args = Setting()
    with open(os.path.join(file_path, 'env_args.txt'), 'a') as f:
        f.write(json.dumps(env_args.__dict__))
    main(args, env_args)
