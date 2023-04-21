from distributions import MultiHeadCategorical
import torch
from utils import init
import torch.nn as nn
from env_setting import Setting
from net import Critic,Actor,ServerActor,Net
from distributions import DiagGaussian
import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Model(nn.Module):
    def __init__(self, state_dim, action_dim, device, trainable=True, hidsize=128,owner='s',num_user=0,bias=20):
        super(Model, self).__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        # w = nn.Linear(hidsize, action_dim)
        # w = init_(w)
        # feature extract
        self.owner = owner
        self.bias = bias
        self.base = MLPBase(num_inputs=state_dim,hidden_size=hidsize)

        # actor
        self.dist = DiagGaussian(hidsize,1,bias=self.bias)
        # self.dist = MultiHeadCategorical(hidsize, 1, action_dim, device)
        env_args = Setting()
        self.scale_par = 1000
        # # critic

        # self.q_network = ServerActor(Net(2, state_dim, device=device), action_dim).to(device)
        self.device = device
        self.identity = torch.eye(action_dim).to(device)
        if trainable:
            self.train()
        else:
            self.eval()

    # @torchsnooper.snoop()
    def act(self, inputs):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        action = dist.sample()

        action_log_probs = dist.log_probs(action)
        return value, action.squeeze(), action_log_probs

    def get_value(self, inputs):
        value,_ = self.base(inputs)

        return value

    def evaluate_actions(self, inputs, action):
        # if self.owner == 's':
        #     action /= torch.tensor(self.scale_par).float()
        value,obs_features = self.base(inputs)

        dist = self.dist(obs_features)

        action_log_probs = dist.log_probs(action)

        dist_entropy = self.dist.entropy().mean()

        return value, action_log_probs, dist_entropy

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
    def __init__(self, num_inputs,recurrent=False,hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs):
        x = inputs


        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)
        debug = self.critic_linear(hidden_critic)

        return self.critic_linear(hidden_critic), hidden_actor