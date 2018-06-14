# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     Simulate_main
   Description :
   Author :       Zhimin Hou
   date：         18-1-12
-------------------------------------------------
   Change Activity:
                   18-1-12
-------------------------------------------------
"""

import os
import argparse
import sys
import time
from baselines import logger
import tensorflow as tf
from baselines.common.misc_util import (set_global_seeds, boolean_flag)
import Simulate_training as training
from models import Actor, Critic
from memory import Memory
from noise import *
from Test_files.EnvPeginHoles import PegintoHoles
import numpy as np
from mpi4py import MPI


# sys.path.append(os.path.abspath('.'))
sys.path.append("/home/zhimin/PycharmProjects/RL_UA/Peg_in_Hole/1-baselines")


# The Successful eposides are run 29, 30, 31
def run(seed, noise_type, layer_norm, **kwargs):
    """Configure things."""
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0: logger.set_level(logger.DISABLED)

    """Create Simulation envs."""
    env = PegintoHoles()

    """Create True envs"""
    # env = Env_robot_control()

    """Parse noise_type"""
    action_noise = None
    param_noise = None
    nb_actions = env.action_dim

    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                        sigma=float(0.2) * np.ones(nb_actions))
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                        sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    """Configure components."""
    memory = Memory(limit=int(1e6), action_shape=env.action_dim, observation_shape=env.state_dim)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)


    """Seed everything to make things reproducible."""
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)

    """Disable logging to avoid noise."""
    start_time = time.time()

    """Train the model"""
    training.train(env=env, param_noise=param_noise,
                   action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)

    """Eval the result"""
    logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    boolean_flag(parser, 'render-eval', default=True)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'restore', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-3) # learning_rate delay[1e-3, 1e-4]
    parser.add_argument('--critic-lr', type=float, default=1e-2) # learning_rate delay[1e-2, 1e-3]
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.95) # 0.99
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=200)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=5) # 20
    parser.add_argument('--nb-train-steps', type=int, default=50)  #  per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--noise-type', type=str,
                        default='adaptive-param_0.15')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    # Run actual script.
    run(**args)