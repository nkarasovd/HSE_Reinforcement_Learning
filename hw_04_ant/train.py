import pybullet_envs
from gym import make
import random
from collections import namedtuple, deque
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam

Transaction = namedtuple('Transaction',
                         ('s', 'a', 'ns', 'r'))

GAMMA = 0.99
TAU = 0.002
CRITIC_LR = 5e-4
ACTOR_LR = 2e-4
BATCH_SIZE = 128
ENV_NAME = "AntBulletEnv-v0"
TRANSITIONS = 1000000


class Buffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.appendleft(Transaction(*args))

    def sample(self, batch_size):
        batch = Transaction(*zip(*random.sample(self.memory, batch_size)))

        return Transaction(s=torch.FloatTensor(batch.s),
                           a=torch.FloatTensor(batch.a),
                           ns=torch.FloatTensor(batch.ns),
                           r=torch.FloatTensor(batch.r))

    def full(self, batch_size):
        if len(self.memory) < batch_size:
            return False
        return True


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


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1))


class RandomNoise:
    def __init__(self, mu=0.0, theta=0.1, sigma=0.1):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dim = 1
        self.low = -1.0
        self.high = 1.0
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state


class DDPG:
    def __init__(self, state_dim, action_dim, low, high, eps, mem_sz):
        self.low = low
        self.high = high
        self.eps = eps

        self.memory = Buffer(mem_sz)
        self.noise = RandomNoise()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.critic_criterion = nn.MSELoss()

    def update(self):
        if not self.memory.full(BATCH_SIZE):
            return

        batch = self.memory.sample(BATCH_SIZE)

        with torch.no_grad():
            na = self.actor(batch.ns)

        temp_1 = self.critic(batch.s, batch.a)
        temp_2 = self.critic_target(batch.ns, na)
        temp_3 = batch.r.unsqueeze(1) + GAMMA * temp_2.detach()

        critic_loss = self.critic_criterion(temp_1, temp_3)
        actor_loss = -self.critic(batch.s, self.actor(batch.s)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()

        self.critic_optim.zero_grad()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.7)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.7)

        self.actor_optim.step()
        self.critic_optim.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))

    def act(self, state):
        with torch.no_grad():
            action = np.clip(self.actor(state).view(-1)
                             + self.eps * self.noise.noise(), self.low, self.high)
        return action.numpy()


if __name__ == "__main__":
    env = make(ENV_NAME)

    # random seed
    seed = 123
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    ddpg = DDPG(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
                low=env.action_space.low[0], high=env.action_space.high[0],
                eps=1.0, mem_sz=60000)

    best_reward = 800
    rewards = []
    score = deque(maxlen=5)

    for i in range(TRANSITIONS):
        state = env.reset()
        total_reward, done = 0, False

        while not done:
            action = ddpg.act(torch.tensor(state, dtype=torch.float).unsqueeze(0))
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            ddpg.memory.push(state, action, next_state, reward)
            state = next_state
            ddpg.update()

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(ddpg.actor.state_dict(), "agent.pkl")

        ddpg.eps *= 0.95
        rewards.append(total_reward)

        info = '{}) Episode reward: {} || Total mean {} || Best reward {}'

        print(info.format(i, total_reward, np.mean(rewards), best_reward))
