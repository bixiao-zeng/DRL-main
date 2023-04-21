import numpy as np
import torch
from storage import RolloutStorage
from model import Model
import torch.optim as optim
import math
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
from  torch import nn
import pickle

class Env(object):
    def __init__(self,args, device, env_args,
                ):
        #========PPO-server===============
        self.args = args
        self.user_num = self.args.user_num
        self.action_std = args.action_std_init
        self.local_ppo_model = Model(self.args.state_dim_s, self.args.action_dim_s, device,owner='s',action_std_init=self.action_std)
        # self.optimizer = optim.Adam(list(self.local_ppo_model.parameters()), lr=self.args.lr_s)
        self.optimizer = torch.optim.Adam([
            {'params': self.local_ppo_model.actor.parameters(), 'lr': args.lr_actor},
            {'params': self.local_ppo_model.critic.parameters(), 'lr': args.lr_critic}
        ])
        self.rollout = RolloutStorage(self.args.D, self.args.mini_batch_num, self.args.state_dim_s,owner='s')
        self.rollout.to(device)
        self.use_gae = self.args.use_gae
        self.gamma = self.args.gamma
        self.gae_param = self.args.gae_param
        self.ppo_epoch = self.args.ppo_epoch
        self.clip = self.args.clip
        self.value_coeff = self.args.value_coeff
        self.clip_coeff = self.args.clip_coeff
        self.ent_coeff = self.args.ent_coeff
        self.datasize = env_args.V['datasize'][:self.user_num]
        #=================================
        self.device = device

        self.action_dim = self.args.user_num
        self.state_dim_s = self.args.state_dim_s
        self.state_dim_c = self.args.state_dim_c

        relationship = env_args.V['relationship']
        self.unit_cost = torch.tensor(env_args.V['cost'][:self.args.user_num])
        self.uni_quality = torch.tensor(env_args.V['uni_quality'][:self.args.user_num])
        # self.uni_quality = np.exp(self.uni_quality)/np.sum(np.exp(self.uni_quality))
        self.A = torch.div(self.unit_cost,self.uni_quality).sum()
        self.nash_useraction = torch.zeros(self.args.user_num)
        self.nash_userreward = torch.zeros(self.args.user_num)
        self.nash_reward = 0
        self.rewards = np.zeros(self.args.user_num)
        self.battery_budget = env_args.V['user_battery_budget']
        self.task_num = env_args.V['task_num']
        self.task_budget = env_args.V['R']
        prob = env_args.V['prob']
        # self.check_cost()

        self.prob = prob
        self.remain_budget = self.args.remain_budget
        self.remain_data = [1000]*self.args.user_num

        self.relationship = np.zeros((self.args.user_num, self.args.user_num))
        for i in range(self.args.user_num):
            for j in range(self.args.user_num):
                self.relationship[i][j] = relationship[i][j]


        self.server_reward = 0
        self.total_server_reward = []
        self.lamda = 10
        self.omega = 1
        self.epsilon = 0.
        self.al = 1
        self.bt = 1

        self.R = 0
        self.task_index = 0
        self.epoch = 0
        self.total_obtain_sensing_data = 0
        # self.max_completion_ratio = 0

        self.complete_task = 0
        self.total_task = 0

        self.task_cnt = np.zeros(self.task_num)
        self.obtain_sensing_data = np.zeros(self.task_num)

        self.final_contrib_data = 0

        self.intrinsic_reward = 0
        self.extrinsic_reward = 0
        # self.nash_computing()
        self.font1 = {'family': 'Times New Roman',
                      'weight': 'normal',
                      'size': 85,
                      }  # label
        self.color = ['purple', 'red', 'green', 'blue', 'orange', 'cyan', 'grey']
        self.ticksize = 80
        self.legfont = {'family': 'Times New Roman',
                        'weight': 'normal',
                        'size': 60,
                        }
        self.nash_serverAct()
        self.nash_computing()
        if not torch.ge(self.nash_useraction,0).all() and torch.ge(self.nash_userreward, 0).all():
            self.nash_optimization()
        self.act_c = 0
        self.act_s = 0
        self.MseLoss = nn.MSELoss()



    def clamp_number(self,num,a,b):
        return max(min(num,max(a,b)),min(a,b))

    def check_cost(self):
        # since raw problem has transformed into KKT problem, now the optimal dn is not merely the solution of derivative u
        checkbound = self.A/(self.user_num-1)
        #make sure the nash action is positive
        for i in range(self.user_num):
            bd = self.uni_quality[i] * checkbound
            if self.unit_cost[i]>=bd:
                print('User %d provided illegal cost of %f, suggesting below %f'%(i,self.unit_cost[i],bd))

    def norm_action(self):
        k = self.datasize/(self.nash_useraction.max()-self.nash_useraction.min())
        self.nash_useraction = k*(self.nash_useraction-self.nash_useraction.min())
        debug = True
    def clip_action(self,idx):
        a,r = self.nash_useraction[idx], self.nash_action,
        b1 = r/self.unit_cost[idx]/self.omega
        b2 = self.datasize[idx]
        b_min = min(b1,b2)
        a = self.clamp_number(a,0,b_min)
        return a

    def nash_serverAct(self):
        x_lst = []
        op_lst = []
        for i in range(self.user_num):
            x = (self.user_num - 1) / self.A
            x_lst.append(x)
            op_lst.append(x * (1 - x * self.unit_cost[i] / self.uni_quality[i]))
        debug = sum(op_lst)
        self.nash_action = self.lamda * self.al - 1 / (sum(op_lst) * self.bt)
        debug = True

    def phi_func(self,a,b,eps=1e-16):
        b = b.transpose(0,1)
        temp = torch.add(torch.pow(a,2),torch.pow(b,2))
        temp = torch.sqrt(temp+eps)
        rst = temp-torch.add(a,b)
        return rst

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.local_ppo_model.set_action_std(self.action_std)
        print("--------------------------------------------------------------------------------------------")

    # def constrain_cmp(self,i):
    #     sensing_data = torch.mul(self.uni_quality, self.nash_useraction).sum()
    #     sense_i = self.uni_quality[i] * self.nash_useraction[i]
    #     self.constrain_grad[i][0] = self.uni_quality[i] * (sensing_data - sense_i) * self.nash_action
    #     self.constrain_grad[i][0] /= torch.pow(sensing_data, 2)
    #     self.constrain_grad[i][0] += self.unit_cost[i]
    #     self.constrain_grad[i][1] = (-1) * torch.ones(self.user_num)
    #     self.constrain_grad[i][2] = torch.ones(self.user_num)
    #     self.constrain[i][0] = -sense_i * self.nash_action / sensing_data
    #     self.constrain[i][0] += self.unit_cost[i] * self.nash_useraction[i]
    #     self.constrain[i][1] = 1 - self.nash_useraction[i]
    #     self.constrain[i][2] = self.nash_useraction[i] - self.datasize[i]

    def constrain_cmp2(self):
        sense_i = torch.mul(self.uni_quality, self.nash_useraction)
        sensing_data = sense_i.sum()
        sense_other = sensing_data - sense_i
        sense_other_nash_server = sense_other * self.nash_action
        constrain_grad = torch.rand(self.st,self.user_num)
        constrain_grad[0] = (-1)*torch.mul(self.uni_quality,sense_other_nash_server)
        constrain_grad[0] /= torch.pow(sensing_data,2)
        constrain_grad[0] = torch.add(constrain_grad[0],self.unit_cost)
        constrain_grad[1] = (-1)*torch.ones(self.user_num)
        # constrain_grad[2] = torch.ones(self.user_num)
        constrain = torch.rand(self.st,self.user_num)
        constrain[0] = sense_i*self.nash_action/sensing_data
        constrain[0] = (-1)*torch.sub(constrain[0],torch.mul(self.unit_cost,self.nash_useraction))
        constrain[1] = (-1)*self.nash_useraction
        # constrain[2] = torch.sub(self.nash_useraction,torch.tensor(self.datasize))
        self.constrain_grad = constrain_grad
        self.constrain = constrain

    def u_function(self,vbose=True):
        # self.nash_useraction = torch.tensor([0.8912125825881958,0.4641661047935486])
        # self.nash_action = torch.tensor(4.166433334350586)
        sense_i = torch.mul(self.uni_quality, self.nash_useraction)
        sensing_data = sense_i.sum()
        if sensing_data<0:
            print('Raise Error: sensing data less than 0!')
        else:
            u = sense_i * self.nash_action / sensing_data
            u = torch.sub(u, torch.mul(self.unit_cost, self.nash_useraction))
            self.nash_userreward = u
            self.nash_reward = self.g_concav(sensing_data) * self.lamda - self.nash_action
            if vbose:
                print('nash_user:', self.nash_useraction, self.nash_userreward)
                print('nash_server:', self.nash_action, self.nash_reward)
        # self.nash_reward = self.g_concav(self.nash_useraction.sum()) * self.lamda - self.nash_action

    def nash_optimization(self,lr=1e-3):
        self.st = 2
        torch.autograd.set_detect_anomaly(True)
        self.nash_useraction = Variable(torch.rand(self.user_num), requires_grad=True)
        self.multiplier = Variable((-1)*torch.rand(self.user_num,self.st),requires_grad=True)
        # self.constrain_grad = torch.rand(self.user_num,3)
        # self.constrain = torch.rand(self.user_num,3)
        self.uni_quality = torch.tensor(self.uni_quality)
        self.unit_cost = torch.tensor(self.unit_cost)
        optimizer = optim.Adam([self.nash_useraction,self.multiplier],lr=lr)
        optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=[40], gamma=0.1)
        # optim.lr_scheduler.StepLR(optimizer, 40, gamma=0.1, last_epoch=-1)
        L_grad = torch.rand(self.user_num)
        u_grad = torch.rand(self.user_num)
        PHI = torch.rand(self.user_num,3)
        num_epochs = 800
        LOSS = []
        min_loss = 1e8
        for epoch in range(num_epochs):
            sense_i = torch.mul(self.uni_quality, self.nash_useraction)
            sensing_data = sense_i.sum()
            sense_other = sensing_data-sense_i
            sense_other_nash_server = sense_other*self.nash_action
            u_grad = torch.mul(self.uni_quality,sense_other_nash_server)
            u_grad /= torch.pow(sensing_data,2)
            u_grad = torch.sub(u_grad,self.unit_cost)
            self.constrain_cmp2()
            L_grad = torch.mm(self.multiplier,self.constrain_grad)
            L_grad = torch.diag(L_grad)
            L_grad = torch.add(u_grad,L_grad)
            PHI = self.phi_func((-1)*self.multiplier,(-1)*self.constrain)
            PHI = torch.flatten(PHI)
            loss = torch.pow(L_grad, 2).sum() + torch.pow(PHI, 2).sum()
            loss = loss / 2
            loss.backward()
            optimizer.step()
            print(epoch,loss.item())
            LOSS.append(loss.item())
            with torch.no_grad():
                self.u_function(vbose=False)
                condi1 = torch.le(self.multiplier,0).all()
                condi2 = torch.ge(self.nash_useraction,0).all()
                condi3 = torch.ge(self.nash_userreward,0).all()
                # condi4 = torch.le(self.nash_useraction,torch.tensor(self.datasize)).all()
            if condi2 and condi3:
                print('CATCH!')
                # break
            if LOSS[-1]<min_loss:
                bst_nash_useraction = self.nash_useraction
                bst_utility = self.nash_userreward
                best_epoch = epoch
                min_loss = LOSS[-1]

        self.nash_useraction = bst_nash_useraction
        self.nash_userreward = bst_utility
        torch.save(bst_nash_useraction,os.path.join('log','nash_uaction_ep=%d_ls=%f.pt'%(best_epoch,min_loss)))
        torch.save(bst_utility,os.path.join('log','utility_ep=%d_ls=%f.pkl'%(best_epoch,min_loss)))
        print('nash_user:',self.nash_useraction,bst_utility)
        self.nash_reward = self.g_concav(self.nash_useraction.sum()) * self.lamda - self.nash_action
        print('nash_server:',self.nash_action,self.nash_reward)
        plt.figure(num=1, figsize=(27,30))
        p = plt.plot(range(num_epochs), LOSS, lw=8)
        plt.xlabel('Epochs', self.font1)
        plt.ylabel('Loss', self.font1)
        plt.title('Loss under lr=%f'%(lr),self.font1)
        plt.xticks(rotation=0, size=self.ticksize)
        plt.yticks(size=self.ticksize)
        plt.tight_layout()
        plt.savefig('log/' + 'Loss.png')
        plt.show()

        debug = True



    def nash_computing(self):
        term0 = (self.user_num - 1)/self.A
        term0 = term0 / self.uni_quality
        term1 = term0*self.nash_action
        term2 = torch.mul(term0,self.unit_cost)
        term2 = 1-term2
        self.nash_useraction = torch.mul(term1,term2)
        self.u_function()

        debug = True


    def act(self, obs):
        value, action, action_log_probs = self.local_ppo_model.act(obs)
        # action = torch.tensor(self.nash_action)
        # action = torch.clip(action,0,self.remain_budget)
        self.R = action

        return value,action,action_log_probs


    def insert(self, obs, action, action_log_probs, value, reward):
        self.rollout.insert(obs, action, action_log_probs, value, reward)

    def after_update(self, obs):
        self.rollout.after_update(obs)

    def load_model(self, path, device, test_device):
        self.local_ppo_model.load_state_dict(torch.load(path, map_location={device: test_device}))


    def update(self):
        beta = 0.2
        with torch.no_grad():
            next_value = self.local_ppo_model.get_value(self.rollout.obs[-1:])

        self.rollout.compute_returns(next_value.detach(), self.use_gae, self.gamma, self.gae_param)

        advantages = self.rollout.returns[:-1] - self.rollout.value_preds[:-1]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        av_value_loss = 0
        av_policy_loss = 0
        av_ent_loss = 0
        av_total_loss = 0
        loss_cnt = 0

        for pp_ep in range(self.ppo_epoch):
            # print('server pp_ep %d====================='%pp_ep)
            batch = 0
            data_generator = self.rollout.feed_forward_generator(advantages)
            for samples in data_generator:

                # signal_init = traffic_light.get()
                torch.cuda.empty_cache()
                old_states, next_obs_batch, old_actions, old_values, return_batch, masks_batch, \
                old_logprobs, advantages_batches = samples
                logprobs, state_values, dist_entropy = self.local_ppo_model.evaluate_actions(old_states, old_actions)

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss
                surr1 = ratios * advantages_batches
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages_batches
                action_loss = -torch.min(surr1, surr2)
                value_loss = 0.5 * self.MseLoss(state_values, return_batch)
                ent_loss = -self.args.ent_coeff  * dist_entropy
                # final loss of clipped objective PPO
                loss = action_loss + value_loss + ent_loss

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                av_value_loss += float(value_loss.mean())
                av_policy_loss += float(action_loss.mean())
                av_ent_loss += float(ent_loss.mean())
                av_total_loss += float(loss.mean())
                loss_cnt += 1
                batch += 1

        return av_value_loss / loss_cnt, av_policy_loss / loss_cnt,av_ent_loss/loss_cnt, av_total_loss/loss_cnt

    def get_collected_data(self):
        return self.final_contrib_data

    def close(self):
        return None

    def reset_server_reward(self):
        server_reward = self.server_reward
        self.server_reward = 0
        return server_reward

    def plot_complete_ratio(self, episode):
        obtain_sensing_data_list = []
        for i in range(self.task_num):
            obtain_sensing_data = self.obtain_sensing_data[i] / self.task_cnt[i]
            obtain_sensing_data_list.append(obtain_sensing_data)
        self.total_obtain_sensing_data = 0
        self.epoch = 0
        self.obtain_sensing_data = np.zeros(self.task_num)
        self.task_cnt = np.zeros(self.task_num)

    def reset(self,owner='s'):
        if owner == 's':
            dim = self.state_dim_s
            # mean_t = torch.tensor([3 if i % (self.args.user_num + 1) == 0 else 0.5 for i in range(dim)])
        else:
            dim = self.state_dim_c
            # mean_t = torch.tensor([3 if i==0 else 0.5 for i in range(dim)])
        state = torch.rand(1,dim)
        # state = torch.normal(mean=mean_t, std=torch.tensor([0.2]*dim))
        # state = state.reshape(1,-1)
        return state


    def get_completion_ratio(self):
        completion_ratio = self.complete_task / self.total_task
        self.complete_task = 0
        self.total_task = 0
        return completion_ratio

    def get_reward(self):
        extrinsic_reward = self.extrinsic_reward / self.user_num
        intrinsic_reward = self.intrinsic_reward / self.user_num
        self.extrinsic_reward = 0
        self.intrinsic_reward = 0
        return extrinsic_reward, intrinsic_reward

    def log_base1p(self,a,N):
        rst = np.log(N+1)/np.log(a)
        return rst

    def g_concav(self,val):
        # val = self.log_base1p(1+self.epsilon,val)
        val = self.al * math.log(1 + self.bt * val)
        return val

    def tensor_append(self,ts,a):
        ts = list(ts)
        if a.ndim > 0:
            ts += list(a)
        else:
            ts.append(a)
        ts = torch.tensor(ts)
        return ts

    def tensor_anycat(self,t1,t2):
        t1,t2 = list(t1),list(t2)
        t1 += t2
        t1 = torch.tensor(t1)
        return t1

    def step_stage1(self, state_s,state_c):
        self.act_s += 1
        state_c[0,0] = self.R
        state_s = state_s.reshape(-1,)
        state_s = self.tensor_append(state_s,self.R)
        state_s = state_s[1:]
        state_s = state_s.reshape(1,-1)
        return state_s,state_c

    def step_stage2(self, action,state_s,state_c):
        print('==========game ',self.act_c,'=========')
        self.act_c += 1
        total_sensing = action.sum()

        reward = torch.zeros(self.user_num)

        for i in range(self.user_num):
            reward[i] = action[i] / total_sensing * self.R  - self.unit_cost[i] * action[i]
            print('user %d action: %f utility: %f' % (i, action[i].item(), reward[i].item()))
        self.rewards = reward
        state_c = state_c.reshape(-1,)
        state_c = self.tensor_append(state_c,action)
        state_c = self.tensor_anycat(state_c[0:1],state_c[self.user_num+1:])
        state_c = state_c.reshape(1,-1)
        if self.act_c == self.args.D:
            self.act_c = 0
            self.server_reward_cur = self.g_concav(total_sensing) * self.lamda - self.R
            self.server_reward += self.server_reward_cur
            state_s = state_s.reshape(-1,)
            state_s = self.tensor_append(state_s,action)
            state_s = state_s[self.user_num:]
            state_s = state_s.reshape(1,-1)

        return state_s,state_c, reward