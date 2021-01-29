from gym import make
import numpy as np
import random
from typing import Tuple, List

GAMMA = 0.98
GRID_SIZE = 50
GRID_SIZE_X = 50
GRID_SIZE_Y = 50
ALPHA = 0.35


def transform_state(state: np.ndarray, _min: np.ndarray, _max: np.ndarray) -> np.ndarray:
    """
    Min-Max scaler.
    :param state: Пара (x, v), где x – координата,
    v - скорость со знаком.
    :param _min: Минимальные значения, которые принимают
    измерения состояния.
    :param _max: Максимальные значения, которые принимают
    измерения состояния.
    :return: Дискретизированная пара (x_new, v_new).
    """
    return ((state - _min) / (_max - _min) * GRID_SIZE).astype(int)


class QLearning:
    def __init__(self, state_dim: Tuple, action_dim: int = 3):
        """
        Зададим Q-table.
        :param state_dim: Пара дискретизированных значений (x_max, v_max),
        которые может принимать координата тележки и скорость.
        :param action_dim: Число возможных действий.
        Толкать машину налево, ничего не делать, толкать машину направо.
        """
        coord, velocity = state_dim[0], state_dim[1]
        self.qlearning_estimate = np.zeros((coord, velocity, action_dim)) + 2.

    def update(self, transition: Tuple):
        """
        Принимает на вход кортеж transition, который содержит
        state, action, next_state, reward, done.
        :param transition: Кортеж информации.
        :return: Обновляет данные в Q-table.
        """
        state, action, next_state, reward, done = transition

        difference = reward + GAMMA * np.max(self.qlearning_estimate[next_state[0], next_state[1], :]) - \
                     self.qlearning_estimate[state[0], state[1], action]

        self.qlearning_estimate[state[0], state[1], action] = \
            (1 - ALPHA) * self.qlearning_estimate[state[0], state[1], action] \
            + ALPHA * difference

    def act(self, state: np.ndarray):
        """
        Совершает действие с максимальным значением
        в Q-table[coord, velocity, :].
        :param state: Состояние, пара (coord, velocity).
        :return: Действие, которое необходимо выполнить.
        """
        coord, velocity = state
        return np.argmax(self.qlearning_estimate[coord, velocity, :])

    def save(self, path: str = "agent.npz"):
        """
        Сохраняем агента.
        :param path:
        :return:
        """
        np.savez(path, self.qlearning_estimate)


def evaluate_policy(agent: QLearning, episodes: int = 5) -> List:
    """
    Оценка стратегии агента.
    :param agent: Агент, экземпляр класса QLearning.
    :param episodes: Число эпизодов.
    :return: Список полученных наград.
    """
    env = make("MountainCar-v0")
    _min, _max = env.observation_space.low, env.observation_space.high
    returns = []
    for _ in range(episodes):
        state = env.reset()
        total_reward, done = 0.0, False

        while not done:
            state, reward, done, _ = env.step(agent.act(transform_state(state, _min, _max)))
            total_reward += reward
        returns.append(total_reward)

    return returns


if __name__ == '__main__':
    env = make("MountainCar-v0")
    _min, _max = env.observation_space.low, env.observation_space.high

    # Зафиксируем seed
    env.seed(42)
    random.seed(42)
    np.random.seed(42)

    ql = QLearning(state_dim=(GRID_SIZE_X, GRID_SIZE_Y))
    eps = 0.1
    transitions = 4000000
    trajectory = []
    state = env.reset()
    s_vel = abs(state[-1])
    state = transform_state(state, _min, _max)

    for i in range(transitions):
        steps = 0

        # Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = ql.act(state)

        # Совершили выбранное действие
        next_state, reward, done, _ = env.step(action)

        # Запомнили скорость, дискретизировали состояния
        ns_vel = abs(next_state[-1])
        next_state = transform_state(next_state, _min, _max)

        # Обновили награду
        reward += (ns_vel - s_vel) * 10

        # Обновили Q-table
        ql.update((state, action, next_state, reward, done))

        # Если эпизод закончился, то идем в начальное состояние,
        # иначе, переходим в новое состояние
        if done:
            state = env.reset()
            s_vel = abs(state[-1])
            state = transform_state(state, _min, _max)
            eps *= 0.9
        else:
            state = next_state

        # Посмотрим, как хорошо гоняет тележка
        if (i + 1) % (transitions // 100) == 0:
            rewards = evaluate_policy(ql)
            print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            ql.save()
