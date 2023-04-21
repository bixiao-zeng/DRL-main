import torch
from storage import RolloutStorage
from model import Model
import torch.optim as optim
import torch.nn as nn

class PPOAgent():
    def __init__(self, args, device):
        self.args = args
        self.bias = self.args.client_bias
        self.action_std =args.action_std_init
        self.local_ppo_model = Model(self.args.state_dim_c-self.args.L, self.args.action_dim_c,device,action_std_init=self.action_std,owner='c')
        # self.optimizer = optim.Adam(list(self.local_ppo_model.parameters()), lr=self.args.lr)
        debg = self.args.state_dim_c-self.args.L
        self.optimizer = torch.optim.Adam([
            {'params': self.local_ppo_model.actor.parameters(), 'lr': args.lr_actor},
            {'params': self.local_ppo_model.critic.parameters(), 'lr': args.lr_critic}
        ])
        self.rollout = RolloutStorage(self.args.D, self.args.mini_batch_num,self.args.state_dim_c-self.args.L ,owner='c')
        self.rollout.to(device)
        self.use_gae = self.args.use_gae
        self.gamma = self.args.gamma
        self.gae_param = self.args.gae_param
        self.ppo_epoch = self.args.ppo_epoch
        self.clip = self.args.clip
        self.value_coeff = self.args.value_coeff
        self.clip_coeff = self.args.clip_coeff
        self.ent_coeff = self.args.ent_coeff
        self.MseLoss = nn.MSELoss()

    def clamp_number(self,num,a,b):
        return max(min(num,max(a,b)),min(a,b))


    def act(self, obs):
        value, action, action_log_probs = self.local_ppo_model.act(obs)
        return value, action, action_log_probs

    def insert(self, obs, action, action_log_probs, value, reward):
        self.rollout.insert(obs, action, action_log_probs, value, reward)

    def after_update(self, obs):
        self.rollout.after_update(obs)

    def load_model(self, path, device, test_device):
        self.local_ppo_model.load_state_dict(torch.load(path, map_location={device: test_device}))

    def reset(self, obs):
        self.rollout.reset(obs)

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
                ent_loss = -self.args.ent_coeff * dist_entropy
                # final loss of clipped objective PPO
                loss = action_loss + value_loss + ent_loss

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                # =============================================
                # obs_batch, next_obs_batch, action_batch, old_values, return_batch, masks_batch, \
                # old_action_log_probs, advantages_batch = samples
                # cur_values, cur_action_log_probs,dis_entropy= self.local_ppo_model.evaluate_actions(obs_batch,action_batch)
                #
                # # ----------use ppo self.args.clip to compute loss------------------------
                # ratio = torch.exp(cur_action_log_probs - old_action_log_probs)
                # surr1 = ratio * advantages_batch
                # surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantages_batch
                #
                # action_loss = -torch.min(surr1, surr2).mean()
                #
                # value_pred_clipped = old_values + (cur_values - old_values).clamp(-self.clip, self.clip)
                # value_losses = (cur_values - return_batch).pow(2)
                # value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                # value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                # # value_loss = torch.mean((return_batch - cur_values)**2)
                #
                # value_loss = value_loss * self.value_coeff
                # action_loss = action_loss * self.clip_coeff
                # # ent_loss = dist_entropy * self.ent_coeff
                # # ------------------ for curiosity driven--------------------------
                # total_loss = value_loss + action_loss
                # # print('batch %d the total loss: %f'%(batch,total_loss))
                # self.local_ppo_model.zero_grad()
                # self.optimizer.zero_grad()
                # total_loss.backward()
                # self.optimizer.step()

                av_value_loss += float(value_loss.mean())
                av_policy_loss += float(action_loss.mean())
                av_ent_loss += float(ent_loss.mean())
                av_total_loss += float(loss.mean())
                loss_cnt += 1
                batch += 1

        return av_value_loss / loss_cnt, av_policy_loss / loss_cnt, av_ent_loss / loss_cnt, av_total_loss / loss_cnt
    # def update(self):
    #     beta = 0.2
    #     with torch.no_grad():
    #         next_value = self.local_ppo_model.get_value(self.rollout.obs[-1:])
    #
    #     self.rollout.compute_returns(next_value.detach(), self.use_gae, self.gamma, self.gae_param)
    #
    #     advantages = self.rollout.returns[:-1] - self.rollout.value_preds[:-1]
    #
    #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #
    #     av_value_loss = 0
    #     av_policy_loss = 0
    #     av_ent_loss = 0
    #     loss_cnt = 0
    #
    #     for pp_ep in range(self.ppo_epoch):
    #         batch = 0
    #         # print('client pp_ep %d=====================' % pp_ep)
    #         data_generator = self.rollout.feed_forward_generator(advantages)
    #         for samples in data_generator:
    #             # signal_init = traffic_light.get()
    #             torch.cuda.empty_cache()
    #             obs_batch, next_obs_batch, action_batch, old_values, return_batch, masks_batch, \
    #             old_action_log_probs, advantages_batch = samples
    #
    #             cur_values, cur_action_log_probs = self.local_ppo_model.evaluate_actions(obs_batch,
    #                                                                                                    action_batch)
    #
    #             # ----------use ppo self.args.clip to compute loss------------------------
    #             ratio = torch.exp(cur_action_log_probs - old_action_log_probs)
    #             surr1 = ratio * advantages_batch
    #             surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantages_batch
    #
    #             action_loss = -torch.min(surr1, surr2).mean()
    #
    #             value_pred_clipped = old_values + (cur_values - old_values).clamp(-self.clip, self.clip)
    #             value_losses = (cur_values - return_batch).pow(2)
    #             value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
    #             value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
    #             # value_loss = torch.mean((return_batch - cur_values)**2)
    #
    #             value_loss = value_loss * self.value_coeff
    #             action_loss = action_loss * self.clip_coeff
    #             # ------------------ for curiosity driven--------------------------
    #             total_loss = value_loss + action_loss
    #             # print('batch %d the total loss: %f'%(batch,total_loss))
    #             self.local_ppo_model.zero_grad()
    #             self.optimizer.zero_grad()
    #             total_loss.backward()
    #             self.optimizer.step()
    #
    #             av_value_loss += float(value_loss)
    #             av_policy_loss += float(action_loss)
    #             loss_cnt += 1
    #             batch += 1
    #
    #     return av_value_loss / loss_cnt, av_policy_loss / loss_cnt,total_loss/loss_cnt
