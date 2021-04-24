import torch
import torch.nn as nn


device = "cpu"


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.model(state)


class Agent:
    def __init__(self):
        self.model = Actor(28, 8).to(device)
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pickle", map_location=device))

    def act(self, state):
        state = torch.tensor(state).to(device).float().unsqueeze(0)
        action = torch.clamp(self.model(state), -1.0, 1.0)
        return action.cpu().data[0].numpy()

    def reset(self):
        pass
