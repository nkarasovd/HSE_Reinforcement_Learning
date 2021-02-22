from gym import make
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from collections import namedtuple, deque
import random
import copy
from typing import NoReturn, List, Tuple, Union

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 5e-4

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class Buffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args) -> NoReturn:
        self.buffer.appendleft(Transition(*args))

    def sample(self, batch_size: int) -> List:
        return random.sample(self.buffer, batch_size)


class DQN:
    def __init__(self, state_dim, action_dim):
        self.steps = 0  # Do not change
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA

        self.buffer = Buffer(27500)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Torch model
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.target = copy.deepcopy(self.model).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()

    def consume_transition(self, transition: Tuple) -> NoReturn:
        """
        Add transition to a replay buffer.
        :param transition: Tuple
        state, action, next_state, reward, done
        :return:
        Hint: use deque with specified maxlen.
        It will remove old experience automatically.
        """
        state, action, next_state, reward, done = transition

        next_state = self.transform(next_state).float()
        state = self.transform(state).float()
        reward = self.transform(reward).float()
        done = self.transform(done).float()

        self.buffer.push(state, action, next_state, reward, done)

    def sample_batch(self) -> List:
        """
        Sample batch from a replay buffer.
        :return: List (batch)
        """
        return self.buffer.sample(self.batch_size)

    def train_step(self, batch: List) -> NoReturn:
        """
        Use batch to update DQN's network.
        :param batch:
        :return:
        """
        batch = Transition(*zip(*batch))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_action_values = self.model(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0].detach()

        expected = (next_state_values * self.gamma) + reward_batch

        loss = self.loss(state_action_values, expected.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def update_target_network(self) -> NoReturn:
        """
        Update weights of a target Q-network here.
        :return:
        """
        self.target.load_state_dict(self.model.state_dict())

    def act(self, state: np.ndarray) -> torch.Tensor:
        """
        Compute an action.
        :param state: np.ndarray
        :return: torch.Tensor
        """
        state = self.transform(state).float()

        with torch.no_grad():
            return self.model(state).max(1)[1].view(1, 1)

    def transform(self, x: Union[int, bool, np.float64, np.ndarray]) -> torch.Tensor:
        """
        Transform to torch.Tensor
        :param x: Union[int, bool, np.float64, np.ndarray]
        :return: torch.Tensor
        """
        return torch.tensor(x).to(self.device).unsqueeze(0)

    def update(self, transition: Tuple) -> NoReturn:
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self) -> NoReturn:
        torch.save(self.model, "agent.pkl")


def evaluate_policy(agent: DQN, episodes: int = 5) -> List:
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state).item())
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make("LunarLander-v2")

    seed = 42
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    state = env.reset()

    for _ in range(INITIAL_STEPS):
        steps = 0
        action = dqn.transform(env.action_space.sample()).unsqueeze(1)
        next_state, reward, done, _ = env.step(action.item())

        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

    best_reward = -200.0

    for i in range(TRANSITIONS):
        steps = 0

        # Epsilon-greedy policy
        if random.random() < eps:
            action = dqn.transform(env.action_space.sample()).unsqueeze(1)
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action.item())

        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            if np.mean(rewards) > best_reward:
                best_reward = np.mean(rewards)
                dqn.save()
