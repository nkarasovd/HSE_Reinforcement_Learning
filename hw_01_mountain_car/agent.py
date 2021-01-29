from gym import make
import numpy as np
from .train import transform_state


class Agent:
    def __init__(self):
        self.qlearning_estimate = np.load(__file__[:-8] + "/agent.npz")['arr_0']

    def act(self, state):
        env = make("MountainCar-v0")
        _min, _max = env.observation_space.low, env.observation_space.high

        state = transform_state(state, _min, _max)
        coord, velocity = state

        return np.argmax(self.qlearning_estimate[coord, velocity, :])

    def reset(self):
        pass
