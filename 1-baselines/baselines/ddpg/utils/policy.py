import random
import numpy as np
from enum import Enum

"""
DISCRETE
"""

class DiscretePolicy:

  def select_action(self, values):
    pass

  def reset(self):
    pass

class Greedy(DiscretePolicy):

  def select_action(self, values):
    return np.argmax(values)

class EpsilonGreedy(DiscretePolicy):

  def __init__(self, epsilon):
    self.epsilon = epsilon

  def select_action(self, values):
    rand = random.random()

    if rand <= self.epsilon:
      return random.randint(0, len(values) - 1)
    else:
      return np.argmax(values)

class EpsilonGreedyAnneal(DiscretePolicy):

  class Mode(Enum):
    LINEAR = 1

  def __init__(self, mode, fract_iters, max_iters, final_epsilon):

    self.mode = mode
    self.fract_iters = fract_iters
    self.max_iters = max_iters
    self.final_epsilon = final_epsilon
    self.step = 0

  def select_action(self, values):
    action =  max(self.final_epsilon, 1 - self.step / (self.fract_iters * self.max_iters))
    self.step += 1
    return action

"""
CONTINUOUS
"""

class ContinuousPolicy:

    # def add_noise(self, action):
    #     return action

    def reset(self):
        pass

class OrnsteinUhlenbeckNoise(ContinuousPolicy):
  # Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
    def __init__(self, action_dim, theta=0.15, sigma=0.25, mu=0.):
        self.action_dim = action_dim

        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def add_noise(self, action):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        action = action + self.state
        # limit the range of actions
        action[0] = np.clip(action[0], -0.2, 0.2)
        action[1] = np.clip(action[1], -0.2, 0.2)
        action[2] = np.clip(action[2], -0.2, 0.2)
        action[3] = np.clip(action[3], -0.2, 0.2)
        action[4] = np.clip(action[4], -0.2, 0.2)
        action[5] = np.clip(action[5], -0.2, 0.2)
        return action

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

class OrnsteinUhlenbeckNoiseAnneal(ContinuousPolicy):

    class Mode(Enum):
        LINEAR = 1

    def __init__(self, action_dim, fract_iters, max_iters, final_fract, theta=0.15, sigma=0.2, mu=0):
        self.action_dim = action_dim

        self.mu = mu
        self.theta = theta
        self.sigma = sigma

        self.fract_iters = fract_iters
        self.max_iters = max_iters
        self.final_fract = final_fract

        self.state = np.ones(self.action_dim) * self.mu
        self.step = 0
        self.reset()

    def add_noise(self, action):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx

        action = action + self.state * max(self.final_fract, (self.step / (self.fract_iters * self.max_iters)))
        self.step += 1
        return action

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

class GaussianNoiseAnneal(ContinuousPolicy):

    class Mode(Enum):
        LINEAR = 1

    # def __init__(self, action_dim, start_eps, max_iters, final_fract):
    #     self.action_dim = action_dim
    #     # self.mode = mode
    #     # self.fract_iters = fract_iters
    #     self.max_iters = max_iters
    #     self.final_fract = final_fract
    #     self.start_eps = start_eps
    #     self.step = 0
    def __init__(self, action_dim, max_sigma=1.0, min_sigma=0.1, explore_period=1000000):
        self.action_dim = action_dim
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.explore_period = explore_period

    def add_noise(self, action, step):
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, step * 1.0/self.explore_period)
        action = action + np.random.normal(self.action_dim)*sigma

        action[0] = np.clip(action[0], -0.2, 0.2)
        action[1] = np.clip(action[1], -0.2, 0.2)
        action[2] = np.clip(action[2], -0.2, 0.2)
        action[3] = np.clip(action[3], -0.2, 0.2)
        action[4] = np.clip(action[4], -0.2, 0.2)
        action[5] = np.clip(action[5], -0.2, 0.2)
        return action

    def reset(self):
        pass

class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient
        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)