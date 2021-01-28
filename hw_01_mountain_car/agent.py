import numpy as np
from train import transform_state


class Agent:
    def __init__(self):
        self.qlearning_estimate = np.load(__file__[:-8] + "/agent.npz")

    def act(self, state):
        state = transform_state(state)
        return np.argmax(self.qlearning_estimate[state])

    def reset(self):
        pass
