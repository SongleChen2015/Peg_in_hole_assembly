import numpy as np
"""
The basic replaybuffer
"""
class ReplayBuffer:
    def __init__(self, size, state_dim, action_dim):
        self.size = size
        self.next_idx = 0
        self.full = False
        if isinstance(state_dim, tuple):
          shape = (self.size, *state_dim)
        else:
          shape = (self.size, state_dim)

        self.states = np.empty(shape)
        self.actions = np.empty((self.size, action_dim))
        self.rewards = np.empty(self.size)
        self.next_states = np.empty(shape)
        self.done = np.empty(self.size)
        self.safe_or_not = np.empty(self.size)
        self.sensors_bound = np.array([1./100, 1.0/100, 1.0/30., 1.0, 1.0, 1.0,
                             1.0/0.1, 1.0/0.1, 1.0/200., 1.0/180., 1.0/0.05, 1.0/0.05])

    def add(self, item):

        self.states[self.next_idx] = item["states"]
        self.actions[self.next_idx] = item["actions"]
        self.rewards[self.next_idx] = item["rewards"]
        self.next_states[self.next_idx] = item["next_states"]
        self.done[self.next_idx] = item["done"]
        self.safe_or_not[self.next_idx] = item["safe_or_not"]

        if self.next_idx == self.size - 1:
          self.full = True

        self.next_idx = (self.next_idx + 1) % self.size

    """直接进行归一化"""
    def sample(self, size):

        if not self.full:
            idxs = np.random.randint(0, self.next_idx, size=size)
        else:
            idxs = np.random.randint(0, self.size, size=size)

        return {
          "states": np.multiply(self.states[idxs, :], self.sensors_bound),
          "actions": self.actions[idxs, :],
          "rewards": self.rewards[idxs],
          "next_states": np.multiply(self.next_states[idxs, :], self.sensors_bound),
          "done": self.done[idxs]
        }

    """为进行归一化状态"""
    def compute_state_mean_and_std(self):
        if self.full:
            end = self.size
        else:
            end = self.next_idx
        state_mean = np.mean(self.states[:end], axis=0)
        # state_std = np.std(self.states[:end], axis=0)
        # rewards_mean = np.mean(self.rewards[:end])
        # rewards_std = np.std(self.rewards[:end])
        state_max = np.max(self.states[:end], axis=0)
        state_min = np.min(self.states[:end], axis=0)
        reward_max = np.max(self.rewards[:end])
        reward_min = np.min(self.rewards[:end])
        return state_max, state_min, state_mean, reward_max, reward_min

    def normalize_states(self, mean, std, mean_reward, rewards_std):
        if self.full:
            end = self.size
        else:
            end = self.next_idx

        self.states[:end] = (self.states[:end] - mean) / std
        self.next_states[:end] = (self.next_states[:end] - mean) / std
        self.rewards[:end] = (self.rewards[:end] - mean_reward) / rewards_std
  # def normalize_linear(self, state_max, state_min, reward_max, reward_min):
  #   if self.full:
  #     end = self.size
  #   else:
  #     end = self.next_idx
  #
  #   self.states[:end] = (self.states[:end] - mean) / std
  #   self.next_states[:end] = (self.next_states[:end] - mean) / std
  #   self.rewards[:end] = (self.rewards[:end] - mean_reward) / rewards_std

    def get_size(self):
        if not self.full:
            return self.next_idx
        else:
            return self.size
