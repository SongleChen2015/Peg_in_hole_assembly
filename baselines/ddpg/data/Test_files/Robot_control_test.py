# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     Robot_control_test
   Description :
   Author :       Zhimin Hou
   date：         18-1-14
-------------------------------------------------
   Change Activity:
                   18-1-14
-------------------------------------------------
"""

import os
import time
from collections import deque
import pickle
import sys
sys.path.append("/home/rvsa/RL_project/Peg_in_Hole/1-baselines")
from baselines import logger
from ddpg import DDPG
from util import mpi_mean, mpi_std, mpi_max, mpi_sum
import baselines.common.tf_util as U
import tensorflow as tf
from mpi4py import MPI
import numpy as np
import pandas as pd
from Env_robot_control import Env_robot_control


epoch_episode_rewards = []
epoch_episode_steps = []
epoch_adaptive_distances = []

epoch_actions = []
epoch_qs = []
epoch_episodes = 0
"""environment reset """
env = Env_robot_control()

episode_reward = 0.
# episode_step = 0
done = False
Force_moments = []
nb_rollout_steps = 200

env.pull_up()

# print(env.robot_control.Tw_h[2, 3])

# if env.reset() is False:
    # exit()

# for t_rollout in range(nb_rollout_steps):
#
#     """scale for execution in env"""
#     action = np.array([0., 0., 0., 0., 0., 0.])
#     new_obs, r, done, info = env.step(action, t_rollout+1)
#     print(new_obs)
#     episode_reward += r
#
#     """Plot the force and moments"""
#     Force_moments.append(new_obs[0:6])
#     print(new_obs[0:6])
#     env.plot_force(Force_moments, t_rollout+1)
#
#     obs = new_obs
#
#     """Episode done and start pull the pegs step by step"""
#     if done:
#         logger.info('Peg-in-hole assembly done!!!')
#         epoch_episode_rewards.append(episode_reward)
#         epoch_episode_steps.append(t_rollout)
#         pull_done = False
#         pull_safe = True
#         while pull_done is False and pull_safe:
#
#             pull_done, pull_safe = env.pull_up() #True env
#
#         if pull_safe is False:
#             logger.info('Pull up the pegs failed for the exceed force!!!')
#             exit()
#
#         break
#
#     """Episode failed and start pull the pegs step by step"""
#     if info is False:
#         logger.info('Assembly failed for the exceed force!!!')
#         pull_done = False
#         pull_safe = True
#         while pull_done is False and pull_safe:
#             pull_done, pull_safe = env.pull_up()  # True env
#
#         if pull_safe is False:
#             logger.info('Assembly failed for the exceed force!!!')
#             exit()
#
#         break
