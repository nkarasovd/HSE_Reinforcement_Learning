import pybullet_envs
from gym import make
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.optim import Adam
import random

ENV_NAME = "Walker2DBulletEnv-v0"

LAMBDA = 0.95
GAMMA = 0.99

ACTOR_LR = 2e-4
CRITIC_LR = 1e-4

CLIP = 0.2
ENTROPY_COEF = 1e-2
BATCHES_PER_UPDATE = 64
BATCH_SIZE = 256

MIN_TRANSITIONS_PER_UPDATE = 4096
MIN_EPISODES_PER_UPDATE = 10

ITERATIONS = 1000


def compute_lambda_returns_and_gae(trajectory):
    lambda_returns, gae = [], []
    last_lr, last_v = 0.0, 0.0

    for _, _, r, _, v in reversed(trajectory):
        ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
        last_lr = ret
        last_v = v
        lambda_returns.append(last_lr)
        gae.append(last_lr - v)

    # Each transition contains state, action, old action probability, value estimation and advantage estimation
    return [(s, a, p, v, adv) for (s, a, _, p, _), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]


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
        # Returns probability of action according to current policy and distribution of actions (use it to compute entropy loss)
        mu = self.model(state)
        sigma = torch.exp(self.sigma).unsqueeze(0).expand_as(mu)
        distr = Normal(mu, sigma)
        return distr.log_prob(action).sum(axis=1), distr

    def act(self, state):
        # Returns an action, not-transformed action and distribution
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        mu = self.model(state)
        sigma = torch.exp(self.sigma).unsqueeze(0).expand_as(mu)
        distr = Normal(mu, sigma)
        pure_action = distr.sample()
        action = torch.tanh(pure_action)
        return action, pure_action, distr


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 1)
        )

    def get_value(self, state):
        return self.model(state)


class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optim = Adam(self.actor.parameters(), ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), CRITIC_LR)

    def update(self, trajectories):
        transitions = sum(trajectories, [])

        state, action, old_prob, target_value, advantage = zip(*transitions)
        state = np.array(state)
        action = np.array(action)
        old_prob = np.array(old_prob)
        target_value = np.array(target_value)
        advantage = np.array(advantage)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(BATCHES_PER_UPDATE):
            idx = np.random.randint(0, len(transitions), BATCH_SIZE)  # Choose random batch
            s = torch.tensor(state[idx]).float()
            a = torch.tensor(action[idx]).float()
            op = torch.tensor(old_prob[idx]).float()  # Probability of the action in state s.t. old policy
            v = torch.tensor(target_value[idx]).float()  # Estimated by lambda-returns
            adv = torch.tensor(advantage[idx]).float()  # Estimated by generalized advantage estimation

            n_p, distr = self.actor.compute_proba(s, a)
            critic_loss = ((v - self.critic.get_value(s).squeeze(1)) ** 2).mean()
            ratio = torch.exp(n_p - op)
            actor_loss = -torch.min(torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv, ratio * adv).mean()

            loss = critic_loss + actor_loss + -distr.entropy().mean() * ENTROPY_COEF

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()

            loss.backward()

            self.actor_optim.step()
            self.critic_optim.step()

    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            action, pure_action, distr = self.actor.act(state)
            prob = distr.log_prob(pure_action).sum(axis=1)
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], prob.cpu().item()

    def save(self):
        torch.save(self.actor.state_dict(), "agent.pkl")


def evaluate_policy(env, agent, episodes=5):
    returns = []

    for _ in range(episodes):
        state = env.reset()
        total_reward, done = 0.0, False

        while not done:
            state, reward, done, _ = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)

    return returns


def sample_episode(env, agent):
    s = env.reset()
    d, trajectory = False, []

    while not d:
        a, pa, p = agent.act(s)
        v = agent.get_value(s)
        ns, r, d, _ = env.step(a)
        trajectory.append((s, pa, r, p, v))
        s = ns

    return compute_lambda_returns_and_gae(trajectory)


if __name__ == '__main__':
    # Random seed
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    env = make(ENV_NAME)
    ppo = PPO(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    _ = env.reset()

    episodes_sampled = 0
    steps_sampled = 0

    best_reward = 100

    for i in range(ITERATIONS):
        trajectories = []
        steps_ctn = 0

        while len(trajectories) < MIN_EPISODES_PER_UPDATE or steps_ctn < MIN_TRANSITIONS_PER_UPDATE:
            traj = sample_episode(env, ppo)
            steps_ctn += len(traj)
            trajectories.append(traj)
        episodes_sampled += len(trajectories)
        steps_sampled += steps_ctn

        ppo.update(trajectories)

        # Estimation and model save
        if (i + 1) % (ITERATIONS // 100) == 0:
            rewards = evaluate_policy(env, ppo, 25)

            mean_reward = np.mean(rewards)

            info = 'Step: {}, Reward mean: {}, Reward std: {}, Episodes: {}, Steps: {}' \
                .format(i + 1, mean_reward, np.std(rewards), episodes_sampled, steps_sampled)
            print(info)

            if mean_reward > best_reward:
                best_reward = mean_reward
                ppo.save()
