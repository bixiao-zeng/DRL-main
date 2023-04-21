import numpy as np
import os
import torch
from environment import Env
from utils import seed_torch
from ppo_agent import PPOAgent
import csv
from torch.utils.tensorboard import SummaryWriter


def model_test(args, env_args, writer):
    seed_torch(args.seed)

    print('in test process')
    os.environ['OMP_NUM_THREADS'] = '1'

    if args.test_device_num == -1:
        test_device_name = 'cpu'

    else:
        test_device_name = 'cuda:' + str(args.test_device_num)
        torch.cuda.set_device(args.test_device_num)

    # -------------get environment information------------
    ppo_agent = []
    for i in range(args.user_num):
        ppo_agent.append(
            PPOAgent(args,test_device_name))

    ori_device_name = 'cuda:' + str(args.device_num)
    model_path = os.path.join(args.root_path, 'model')
    for i, agent in enumerate(ppo_agent):
        ppo_model_path = os.path.join(model_path, 'ppo_model' + str(i) + '.pt')
        agent.load_model(ppo_model_path, ori_device_name, test_device_name)
        agent.local_ppo_model.eval()

    done_time = 0
    episode_length = 0

    user_num = args.user_num
    env = Env(args,test_device_name,env_args)
    server_path = os.path.join(model_path, 'server_model.pt')
    env.local_ppo_model.load_state_dict(torch.load(server_path))
    env.local_ppo_model.eval()
    action = torch.zeros(user_num, dtype=torch.long)
    final_av_reward = 0
    final_av_server_reward = 0
    test_file_path = os.path.join(args.root_path, 'test_file')
    if not os.path.exists(test_file_path):
        os.mkdir(test_file_path)
    test_result_profile = open(test_file_path + '/test_result.csv', 'w', newline='')
    test_writer = csv.writer(test_result_profile)

    av_ext_reward = 0
    av_int_rewards = 0

    av_completion_ratio = 0
    value = torch.zeros(user_num)
    action_log_probs = torch.zeros(user_num)

    result_path = test_file_path + '/test_result.npz'
    # -----------------------------------------

    all_remaining_energy = []
    rewards = []
    while True:
        action_explo_c = np.zeros((user_num, args.exploration_steps))
        contribu_explo_c = np.zeros((user_num, args.exploration_steps))
        action_explo_s = np.zeros(args.exploration_steps)
        utility_explo_c = np.zeros((user_num, args.exploration_steps))
        utility_explo_s = np.zeros(args.exploration_steps)
        if episode_length >= args.max_test_length:
            print('training over')
            break

        print('---------------in episode ', episode_length, '-----------------------')

        step = 0
        done = True
        av_reward_c = 0
        av_reward_s = 0
        av_action = torch.zeros(user_num)
        state_s, state_c = env.reset()

        interact_time = 0
        sum_reward = 0
        while step < args.exploration_steps:

            interact_time += 1
            # ----------------sample actions(no grad)------------------------
            with torch.no_grad():
                server_value, server_action, server_action_logprobs = env.act(state_s)
                action_explo_s[step] = server_action
                state_s, state_c = env.step_stage1(state_s, state_c)
                for i, agent in enumerate(ppo_agent):
                    value[i], action[i], action_log_probs[i] = agent.act(state_c[i])
                    action[i] = env.clip_action(action[i],env.R,i)
                    action_explo_c[i, step] = action[i]
                    contribu_explo_c[i,step] = action[i]*env.uni_quality[i]

                state_s, state_c, reward, done = env.step_stage2(action, state_s, state_c)
            for i in range(args.user_num):
                utility_explo_c[i,step] = reward[i]
            utility_explo_s[step] = env.server_reward
            av_reward_c += np.mean(reward.numpy())
            av_action += 0.2 * action.float()
            sum_reward += reward.numpy().mean()
            step = step + 1

        action_avg_c = np.mean(action_explo_c, axis=1)
        contribu_avg_c = np.mean(contribu_explo_c,axis=1)
        action_avg_s = np.mean(action_explo_s)
        utility_avg_c = np.mean(utility_explo_c, axis=1)
        server_reward = env.reset_server_reward() / args.exploration_steps
        rewards.append(sum_reward/args.exploration_steps)

        contribution_set = np.array([i*600 for i in env.uni_quality])
        contribution_level = contribu_avg_c/contribution_set
        av_reward_c /= args.exploration_steps

        completion_ratio = env.get_completion_ratio()
        av_completion_ratio += completion_ratio

        final_av_reward += av_reward_c
        final_av_server_reward += env.reset_server_reward() / args.exploration_steps
        episode_length += 1
        for i in range(args.user_num):
            writer.add_scalars('Test action/user_'+str(i),{'real':action_avg_c[i],'nash':env.nash_useraction[i]},episode_length)
            writer.add_scalars('Test utility/user_'+str(i),{'real':utility_avg_c[i],'nash':env.nash_userreward[i]},episode_length)
            print('user %s|| real action %s' % (i, action_avg_c[i]))
            print('user %s|| real utility %s' % (i, utility_avg_c[i]))
        writer.add_scalars('Test action/server',{'real':action_avg_s,'nash':env.nash_action},episode_length)
        writer.add_scalars('Test action/client_avg', {'real':action_avg_c.mean(),'nash':env.nash_useraction.mean()}, episode_length)
        writer.add_scalars('Test utility/server',{'real':server_reward,'nash':env.nash_reward},episode_length)
        writer.add_scalars('Test utility/users_avg',{'real':rewards[-1],'nash':env.nash_userreward.mean()},episode_length)

    test_writer.writerow(
        ['vehicle reward', 'server reward', 'extrinsic reward', 'intrinsic reward', 'completion ratio'])
    test_writer.writerow([final_av_reward / args.max_test_length, final_av_server_reward / args.max_test_length,
                          av_ext_reward / args.max_test_length, av_int_rewards / args.max_test_length,
                          av_completion_ratio / args.max_test_length])
    test_result_profile.close()
    np.savez(result_path, np.asarray(all_remaining_energy))
    print('Finish! Results saved in ', args.root_path)

