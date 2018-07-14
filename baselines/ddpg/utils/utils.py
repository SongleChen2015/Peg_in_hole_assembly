import os, math, gym
import numpy as np

def sample(probs):
  return np.random.choice(len(probs), None, p=probs)

def new_summary_dir(summary_dir):
  if not os.path.isdir(summary_dir):
    os.makedirs(summary_dir)

  i = 1
  while os.path.isdir(os.path.join(summary_dir, "run{}".format(i))):
    i += 1

  summary_dir = os.path.join(summary_dir, "run{}".format(i))
  os.mkdir(summary_dir)
  return summary_dir

def log_params(summary_dir, params):

  with open(os.path.join(summary_dir, "params.txt"), "w") as file:
    for key in params.keys():
      file.write("{}: {}\n".format(key, params[key]))

class MonitorCallable:

  def __init__(self, video_ep_freq=100):
    self.video_ep_freq = video_ep_freq

  def call(self, idx):
    if self.video_ep_freq == 0:
      return False
    else:
      return (idx != 0) and (idx % self.video_ep_freq == 0)

class Timer:

  def __init__(self):
    self.times = {}
    self.counts = {}

  def add(self, name, value):

    if name in self.times:
      self.times[name] += value
      self.counts[name] += 1
    else:
      self.times[name] = value
      self.counts[name] = 1

  def reset(self, name):
    self.times[name] = 0
    self.counts[name] = 0

  def reset_all(self):
    for key in self.times.keys():
      self.times[key] = 0
      self.counts[key] = 0

  def print(self):
    for key in self.times.keys():
      print("%s: %.8f" % (key, (self.times[key] / self.counts[key])))

def filtered_env_factory(env):

  env_type = type(env)

  class FilteredEnv(env_type):

    def __init__(self):
      # transfer properties
      self.__dict__.update(env.__dict__)

      # state
      if math.inf not in self.observation_space.high and - math.inf not in self.observation_space.low:
        self.state_mean = (env.observation_space.high + env.observation_space.low) / 2
        self.state_diff = (env.observation_space.high - env.observation_space.low) / 2

        self.observation_space = gym.spaces.Box(self.filter_state(env.observation_space.low),
                                                self.filter_state(env.observation_space.high))
      else:
        self.state_mean = None
        self.state_diff = None

      # actions
      if math.inf not in self.action_space.high and - math.inf not in self.action_space.low:
        self.actions_mean = (env.action_space.high + env.action_space.low) / 2
        self.actions_diff = (env.action_space.high - env.action_space.low) / 2


        self.action_space = gym.spaces.Box(- np.ones_like(env.action_space.high), np.ones_like(env.action_space.high))
      else:
        self.actions_mean = None
        self.actions_diff = None

    def step(self, action):
      filtered_action = np.clip(self.filter_action(action), self.action_space.low, self.action_space.high)

      state, reward, done, info = env_type.step(self, filtered_action)
      filtered_state = self.filter_state(state)

      return filtered_state, reward, done, info

    def filter_state(self, state):
      if self.state_mean is None or self.state_diff is None:
        return state
      else:
        return (state - self.state_mean) / self.state_diff

    def filter_action(self, action):
      if self.actions_mean is None or self.actions_diff is None:
        return action
      else:
        return action * self.actions_diff + self.actions_mean

  return FilteredEnv()