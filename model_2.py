from distributions import MultiHeadCategorical
import torch
from utils import init
import torch.nn as nn
from env_setting import Setting
from net import Net,Actor,ServerActor,Critic
import numpy as np
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Model(nn.Module):
    def __init__(self, state_dim, action_dim, device, trainable=True, hidsize=128,owner='s'):
        super(Model, self).__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        # feature extract
        # self.base = nn.Sequential(
        #     init_(nn.Linear(state_dim, 128)),
        #     nn.ReLU(),
        #     init_(nn.Linear(128, hidsize)),
        #     nn.ReLU()
        # ).to(device)
        # self.dist = MultiHeadCategorical(hidsize, 1, action_dim, device)
        env_args = Setting()

        self.base = Net(2,state_dim,device=device)
        self.scale_par = 1
        if owner == 's':
            self.dist = ServerActor(self.base,(action_dim,)).to(device)
            # self.scale_par = int(env_args.V['R'][0] / action_dim)
        else:
            self.dist = Actor(self.base,(action_dim,)).to(device)
            # self.scale_par = int(env_args.V['datasize'][0] / action_dim)

        # actor

        # # critic
        # self.critic = nn.Sequential(
        #     init_(nn.Linear(hidsize, 1))
        # ).to(device)
        # critic
        # self.q_network = nn.Sequential(
        #     init_(nn.Linear(hidsize, action_dim)),
        # ).to(device)
        self.q_network = Critic(self.base,action_dim).to(device)
        self.device = device
        self.identity = torch.eye(action_dim).to(device)
        if trainable:
            self.train()
        else:
            self.eval()

    # @torchsnooper.snoop()
    def act(self, inputs):
        with torch.no_grad():
            obs_feature = self.base(inputs)

            # value = self.critic(obs_feature)
            # self.dist(obs_feature)
            # action = self.dist.sample()
            # action_log_probs = self.dist.log_probs(action)

            action = self.dist(inputs)
            action_log_probs = torch.log(action)
            action_log_probs = action_log_probs.mean(-1, keepdim=True)

            # q_value = self.q_network(obs_feature)

            q_value = self.q_network(inputs)
            # mean
            # value = torch.sum(self.dist.probs * q_value, -1, keepdim=True)
            value = torch.sum(action * q_value, -1, keepdim=True)
            # action = action.squeeze()*self.scale_par
            action = torch.argmax(action)
            action = action*self.scale_par

        return value, action, action_log_probs

    def get_value(self, inputs):
        obs_feature = self.base(inputs)
        # value = self.critic(obs_feature)
        # self.dist(obs_feature)
        q_value = self.q_network(inputs)
        value = torch.sum(self.dist(inputs) * q_value, -1, keepdim=True)
        return value

    def evaluate_actions(self, inputs, action):
        obs_features = self.base(inputs)
        # value = self.critic(obs_features)
        q_value = self.q_network(inputs)
        action = action.squeeze(-1)/self.scale_par
        see = action.long()
        index = self.identity[action.long()]
        value = torch.sum(q_value * index, -1).unsqueeze(-1)
        action_probs = self.dist(inputs)
        action_log_probs = torch.log(action_probs)
        action_log_probs = torch.mean(action_log_probs,dim=0)

        dist_entropy = -action_probs*torch.log(action_probs)
        dist_entropy = torch.mean(dist_entropy)

        return value, action_log_probs, dist_entropy

    def print_grad(self):
        for name, p in self.named_parameters():
            print('name: ', name, ' value: ', p.grad.mean(), 'p.requires_grad', p.requires_grad)


