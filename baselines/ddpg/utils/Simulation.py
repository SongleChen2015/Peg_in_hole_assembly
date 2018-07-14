import numpy as np

class ContinuousSimulation:

  def __init__(self, env, agent, policy, prep, steps_before_train=100, train_freq=1, drop_end=False):
    self.env = env
    self.agent = agent
    self.policy = policy
    self.prep = prep

    self.steps_before_train = steps_before_train
    self.train_freq = train_freq
    self.drop_end = drop_end

    self.step = 0
    self.learn_step = 0
    self.ep_step = 0

  def run_episode(self, eval_run=False):

    episode_reward = 0
    self.policy.reset()

    state = self.env.reset()
    state, skip = self.prep.process(state)

    while True:

      if skip:
        action = self.env.action_space.sample()
      else:
        action = self.agent.act(state)

        if not eval_run:
          action = self.policy.add_noise(action)

      action = np.clip(action, -1, 1)

      prev_state = state
      prev_skip = skip

      state, reward, done, _ = self.env.step(action)
      state, skip = self.prep.process(state)

      episode_reward += reward


      if not prev_skip and not skip:
        if not self.drop_end or not done:
          transition = {
            "state": prev_state[0],
            "action": action,
            "reward": reward,
            "next_state": state[0],
            "done": int(done)
          }

          self.agent.perceive(transition)

      if not eval_run and self.step >= self.steps_before_train:
        # learn
        for _ in range(self.train_freq):
          self.agent.learn()
          self.learn_step += 1

      self.step += 1

      if done:
        break

    if eval_run:
      tag = "episode_eval_score"
    else:
      tag = "episode_score"

    self.agent.log_scalar(tag, episode_reward, self.ep_step)
    self.ep_step += 1

    return episode_reward

  def eval_avg(self, runs):
    total_score = 0

    for _ in range(runs):
      total_score += self.run_episode(eval_run=True)

    total_score /= runs
    return total_score