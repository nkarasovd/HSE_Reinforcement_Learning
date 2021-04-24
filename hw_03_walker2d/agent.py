import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, action_dim)
        )
        self.sigma = nn.Parameter(torch.zeros(action_dim))

    def compute_proba(self, state, action):
        mu = self.model(state)
        sigma = torch.exp(self.sigma).unsqueeze(0).expand_as(mu)
        distr = Normal(mu, sigma)
        return distr.log_prob(action).sum(axis=1), distr

    def act(self, state):
        mu = self.model(state)
        sigma = torch.exp(self.sigma).expand_as(mu)
        distr = Normal(mu, sigma)
        pure_action = distr.sample()
        action = torch.tanh(pure_action)
        return action, pure_action, distr


class Agent:
    def __init__(self):
        self.model = Actor(22, 6)
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pkl"))
        self.model.eval()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state).float()
            action, _, _ = self.model.act(state)

        return action.cpu().numpy()

    def reset(self):
        pass
