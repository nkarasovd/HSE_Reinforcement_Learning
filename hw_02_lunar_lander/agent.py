import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform(state):
    return torch.tensor(state).to(device).unsqueeze(0)


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl").to(device)
        self.model.eval()

    def act(self, state):
        with torch.no_grad():
            return self.model(transform(state)).max(1)[1].view(1, 1).item()

    def reset(self):
        pass


agent = Agent()
