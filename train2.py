import numpy as np
import os
import torch
from environment import Env
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

def clamp_number(num, a, b):
    return max(min(num, max(a, b)), min(a, b))

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
    episode_length = 0

    user_num = args.user_num
    env = Env(args,device,env_args)

    action = torch.zeros(user_num, dtype=torch.long)
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

    if args.load_model:
        # env.local_ppo_model.load_state_dict(torch.load(server_path))
        # for i, ppo in enumerate(ppo_agent):
        #     ppo_model_path = os.path.join(model_path, 'ppo_model' + str(i) + '.pt')
        #     ppo.local_ppo_model.load_state_dict(torch.load(ppo_model_path))
        model_test(args, env_args,writer)
        sys.exit()
    while True:
        action_explo_c = np.zeros((user_num, args.exploration_steps))
        action_explo_s = np.zeros(args.exploration_steps)
        utility_explo_c = np.zeros((user_num, args.exploration_steps))
        utility_explo_s = np.zeros(args.exploration_steps)
        nsaction_explo_c = np.zeros((user_num, args.exploration_steps))
        nsutility_explo_c = np.zeros((user_num, args.exploration_steps))

        if episode_length >= args.max_episode_length:
            print('training over')
            break

        print('---------------in episode ', episode_length, '-----------------------')

        step = 0
        #
        # 将所有用户的初始决策置0
        state_s, state_c = env.reset()
        env.after_update(state_s)
        for i, agent in enumerate(ppo_agent):
            agent.after_update(state_c[i])
        done = True
        av_reward = torch.zeros(user_num)
        av_action = torch.zeros(user_num)


        interact_time = 0
        sum_reward = 0.0
        sum_user_reward = np.zeros(user_num)
        while step < args.exploration_steps:

            interact_time += 1
            # ----------------sample actions(no grad)------------------------
            with torch.no_grad():
                server_value,server_action,server_action_logprobs = env.act(state_s)
                action_explo_s[step] = server_action
                state_s,state_c = env.step_stage1(state_s,state_c)
                for i, agent in enumerate(ppo_agent):
                    value[i], action[i], action_log_probs[i] = agent.act(state_c[i])
                    action[i] = env.clip_action(i)
                    action_explo_c[i,step] = action[i]
                    # print(action[i])
                state_s,state_c, reward, done = env.step_stage2(action,state_s,state_c)
            for i in range(args.user_num):
                utility_explo_c[i,step] = reward[i]
            utility_explo_s[step] = env.server_reward
            sum_reward += reward.numpy().mean()
            sum_user_reward += reward.numpy()

            av_reward += reward
            av_action += 0.2 * action.float()
            # ---------judge if game over --------------------
            masks = torch.tensor([[0.0] if done else [1.0]])
            # ----------add to memory ---------------------------
            env.insert(state_s.detach(), server_action.detach(), server_action_logprobs.detach(), server_value.detach(),
                             torch.tensor(env.server_reward_cur), masks.detach())
            for i, agent in enumerate(ppo_agent):
                agent.insert(state_c[i].detach(), action[i].detach(), action_log_probs[i].detach(), value[i].detach(),
                             reward[i].detach(), masks.detach())
            step = step + 1

        action_avg_c = np.mean(action_explo_c,axis=1)
        action_avg_s = np.mean(action_explo_s)
        utility_avg_c = np.mean(utility_explo_c,axis=1)
        utility_avg_s = env.server_reward/args.exploration_steps

        rewards.append(sum_reward/args.exploration_steps)
        server_reward = env.reset_server_reward() / args.exploration_steps
        server_rewards.append(server_reward)
        for i in range(args.user_num):
            writer.add_scalars('action/user_'+str(i),{'real':action_avg_c[i],'nash':env.nash_useraction[i]},episode_length)
            writer.add_scalars('utility/user_'+str(i),{'real':utility_avg_c[i],'nash':env.nash_userreward[i]},episode_length)
            print('user %s|| real action %s' % (i, action_avg_c[i]))
            print('user %s|| real utility %s' % (i, utility_avg_c[i]))
        writer.add_scalars('action/server',{'real':action_avg_s,'nash':env.nash_action},episode_length)
        writer.add_scalars('action/client_avg', {'real':action_avg_c.mean(),'nash':env.nash_useraction.mean()}, episode_length)
        writer.add_scalars('utility/server',{'real':server_reward,'nash':env.nash_reward},episode_length)
        writer.add_scalars('utility/users_avg',{'real':rewards[-1],'nash':env.nash_userreward.mean()},episode_length)
        print('server || action %s'%(action_avg_s))
        print('server || utility %s'%(server_reward))

        av_value_loss = 0
        av_policy_loss = 0
        av_ent_loss = 0

        for i in range(user_num):
            user_rewards[i].append(sum_user_reward[i] / args.exploration_steps)

        for i, agent in enumerate(ppo_agent):
            value_loss, policy_loss, ent_loss, total_loss = agent.update(done)
            av_value_loss += value_loss
            av_policy_loss += policy_loss
            av_ent_loss += ent_loss
        av_value_loss /= user_num
        av_policy_loss /= user_num
        av_ent_loss /= user_num
        value_loss_s, policy_loss_s, ent_loss_s,total_loss_s = env.update()

        writer.add_scalars('Loss/value_loss', {'server':value_loss_s,'client':av_value_loss}, episode_length)
        writer.add_scalars('Loss/policy_loss', {'server':policy_loss_s,'client':av_policy_loss}, episode_length)
        writer.add_scalars('Loss/ent_loss', {'server':ent_loss_s,'client':av_ent_loss}, episode_length)
        writer.add_scalars('Loss/total_loss', {'server': total_loss_s, 'client': total_loss}, episode_length)
        av_reward /= args.exploration_steps
        av_action /= args.exploration_steps
        episode_length += 1

    np.savez(result_path, np.asarray(rewards), np.asarray(server_rewards), np.asarray(ext_rewards),
             np.asarray(int_rewards))
    # reward_profile.close()
    torch.save(env.local_ppo_model.state_dict(), server_path)
    for i, ppo in enumerate(ppo_agent):
        ppo_model_path = os.path.join(model_path, 'ppo_model' + str(i) + '.pt')
        torch.save(ppo.local_ppo_model.state_dict(), ppo_model_path)
    model_test(args, env_args,writer)

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
    parser.add_argument('--max-episode-length', type=int, default=1000)
    parser.add_argument('--exploration-steps', type=int, default=100)
    parser.add_argument('--max-test-length', type=int, default=100)
    parser.add_argument('--mini-batch-num', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ppo-epoch', type=int, default=4)
    parser.add_argument('--load_model', type=bool, default=False)
    # ------------------------------------- parameters that never recommend to be changed ---------------------
    parser.add_argument('--lr', type=float, default=0.0003, help='optimizer learning rate')
    parser.add_argument('--lr_s', type=float, default=0.0003, help='optimizer learning rate')
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--ent-coeff', type=float, default=0.01)
    parser.add_argument('--value-coeff', type=float, default=0.1)
    parser.add_argument('--clip-coeff', type=float, default=1.0)
    parser.add_argument('--use-gae', type=bool, default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_param', type=float, default=0.95)
    parser.add_argument('--remain_budget', type=int, default=1e6)
    parser.add_argument('--remain_data', type=int, default=1000)
    parser.add_argument('--action_dim_c',type=int,default=6)
    parser.add_argument('--action_dim_s', type=int, default=6)
    parser.add_argument('--server_bias', type=float, default=80)
    parser.add_argument('--client_bias', type=float, default=5)

    parser.add_argument('--D', type=int, default=20,help='every D actions update model')
    parser.add_argument('--L', type=int, default=5,help='the window size of the state')


    args = parser.parse_args()
    args.state_dim_s = torch.randn((args.user_num+1)*args.L)
    args.state_dim_c = torch.randn((args.user_num-1)*args.L+1)
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
