from distributions import MultiHeadCategorical
import torch
from utils import init
import torch.nn as nn
from env_setting import Setting
from net import Critic,Actor,ServerActor,Net
from distributions import DiagGaussian
from torch.distributions import MultivariateNormal
from TruncatedNormal import TruncatedNormal

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Model(nn.Module):
    def __init__(self, state_dim, action_dim, device, trainable=True, hidsize=6,owner='s',action_std_init=0.200):
        super(Model, self).__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        # init_ = lambda  m:nn.init.xavier_uniform_
        self.action_std_init = action_std_init
        self.action_var = torch.full((action_dim,), self.action_std_init * self.action_std_init)
        self.action_dim = action_dim
        self.action_dim = action_dim
        self.bias=4
        # w = nn.Linear(hidsize, action_dim)
        # w = init_(w)
        # feature extract
        self.owner = owner
        self.actor = nn.Sequential(
            init_(nn.Linear(state_dim, 6)),
            nn.Tanh(),
            init_(nn.Linear(6, hidsize)),
            nn.Tanh(),
            init_(nn.Linear(hidsize, action_dim)),
        ).to(device)

        # actor
        # self.dist = DiagGaussian(hidsize,1)
        # self.dist = nn.Sequential(
        #     init_(nn.Linear(hidsize, action_dim)),
        #     nn.ReLU()
        # ).to(device)
        # self.dist = MultiHeadCategorical(hidsize, 1, action_dim, device)
        # if self.owner == 'c':
        #     self.scale_par = int(env_args.V['datasize'][0]/(action_dim-1))
        # else:
        #     self.scale_par = int(env_args.V['R']/(action_dim-1))

        # # critic
        # self.critic = nn.Sequential(
        #     init_(nn.Linear(hidsize, 1))
        # ).to(device)
        # critic
        self.critic = nn.Sequential(
            init_(nn.Linear(state_dim, 6)),
            nn.Tanh(),
            init_(nn.Linear(6, hidsize)),
            nn.Tanh(),
            init_(nn.Linear(hidsize, action_dim)),
        ).to(device)

        # self.q_network = ServerActor(Net(2, state_dim, device=device), action_dim).to(device)
        self.device = device
        self.identity = torch.eye(action_dim).to(device)
        if trainable:
            self.train()
        else:
            self.eval()

    def set_action_std(self,action_sdt):
        self.action_std_init = action_sdt
        self.action_var = torch.full((self.action_dim,), self.action_std_init * self.action_std_init)

    def act(self, state):
        action_mean = torch.exp(self.actor(state))
        if self.owner=='s':
            action_mean = action_mean+self.bias
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        # action_mean = torch.tensor([[-2.8031]])
        # cov_mat = torch.tensor([[[0.3600]]])
        dist = TruncatedNormal(action_mean, cov_mat,a=0,b=1e8)
        action = dist.sample()
        # while torch.isinf(action).any() or torch.isnan(action).any():
        #     action = torch.tensor([[[0]]])
        print(action_mean,cov_mat,action)
        action_logprob = dist.log_prob(action).squeeze()
        action = action.squeeze()
        return self.critic(state).squeeze(),action.detach(), action_logprob.detach()

    def evaluate_actions(self, state, action):

        action_mean = torch.exp(self.actor(state))
        if self.owner=='s':
            action_mean = action_mean+self.bias
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        dist = TruncatedNormal(action_mean, cov_mat,a=0,b=1e8)
        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist._entropy
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

    def get_value(self, inputs):
        q_value = self.critic(inputs)
        return q_value

    # @torchsnooper.snoop()
    # def act(self, inputs):
    #     with torch.no_grad():
    #         obs_feature = self.actor(inputs)
    #         owner = self.owner
    #         action = self.dist(obs_feature)
    #         action_log_probs = action
    #
    #         q_value = self.q_network(obs_feature)
    #         # mean
    #     return q_value, action.squeeze(), action_log_probs
    #


    # def evaluate_actions(self, inputs, action):
    #     obs_features = self.actor(inputs)
    #     # value = self.critic(obs_features)
    #     q_value = self.q_network(obs_features)
    #     action_log_probs = action
    #     return q_value, action_log_probs

    def print_grad(self):
        for name, p in self.named_parameters():
            print('name: ', name, ' value: ', p.grad.mean(), 'p.requires_grad', p.requires_grad)

class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=200):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs