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

# -*- coding: utf-8 -*-
import os
import time
from collections import deque
import pickle
import sys
from baselines import logger
from simulation_ddpg import DDPG
from util import mpi_mean, mpi_std, mpi_max, mpi_sum
import baselines.common.tf_util as U
import tensorflow as tf
from mpi4py import MPI
import numpy as np
import pandas as pd


"""First the path should be added."""
sys.path.append("/home/zhimin/PycharmProjects/RL_UA/Peg_in_Hole/1-baselines")


def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
          normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
          popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
          tau=0.01, eval_env=None, param_noise_adaption_interval=50, restore=False):
    rank = MPI.COMM_WORLD.Get_rank()
    max_action = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    # min_action = np.array([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2])

    logger.info('scaling actions by {} before executing in env'.format(max_action))
    model_directory = '/home/zhimin/PycharmProjects/RL_UA/Peg_in_Hole/1-baselines/baselines/ddpg/simulation_data'

    agent = DDPG(actor, critic, memory, env.state_dim, env.action_dim,
                 gamma=gamma, tau=tau, normalize_returns=normalize_returns,
                 normalize_observations=normalize_observations,
                 batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
                 actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                 reward_scale=reward_scale, restore=restore)

    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    saver = tf.train.Saver()

    """Set up logging stuff only for a single worker"""
    # if rank == 0:
    #     saver = tf.train.Saver()
    # else:
    #     saver = None
    # eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)

    with U.single_threaded_session() as sess:
        """Prepare everything"""
        if restore:
            saver = tf.train.import_meta_graph(model_directory + 'model.meta')
            agent.restore_model(model_directory, saver, sess)
        else:
            agent.initialize(sess)
            sess.graph.finalize()

        """Agent Reset"""
        agent.reset()
        # episode_step = 0
        # episodes = 0
        # t = 0
        """Force calibration"""
        # if env.robot_control.CalibFCforce() is False:
        #     exit()

        delay_rate = np.power(10, 1 / nb_epochs)
        epoch_start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_adaptive_distances = []
        epoch_episodes_discount_reward = []
        epoch_episodes_average_reward = []

        epoch_actions = []
        epoch_qs = []
        Force_moments = []
        epoch_episodes = 0
        Long_term_reward = - 0.10
        for epoch in range(nb_epochs):

            """Show the result for cycle 20 times and Save the model"""
            epoch_actor_losses = []
            epoch_critic_losses = []
            """Delay the learning rate"""
            epoch_actor_lr = actor_lr / delay_rate
            epoch_critic_lr = critic_lr / delay_rate

            for cycle in range(nb_epoch_cycles):
                """environment reset """
                agent.reset()
                obs = env.reset()
                episode_reward = 0.
                episode_discount_reward = 0.
                q_value = 0.
                done = False
                forcement = []
                Last_average_reward = 0.
                Number_episodes = 0.
                for t_rollout in range(nb_rollout_steps):

                    """Predict next action"""
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    assert action.shape[0] == env.action_dim

                    q_value += q
                    """scale for execution in env"""
                    new_obs, r, done, info, expert_action = env.step(action, t_rollout)
                    episode_discount_reward += gamma * r

                    """adapt_action_noise"""
                    agent.feed_back_explore(action, expert_action)

                    logger.info("The maximum force:" + str(max(abs(new_obs[0:3]))) + " The maximum moments:" + str(max(abs(new_obs[3:6]))))
                    episode_reward += r

                    delta = r - Long_term_reward
                    # if memory.nb_entries >= batch_size and param_noise is not None:
                    #     agent.feed_back_explore(delta)
                    Number_episodes = gamma + gamma*Number_episodes
                    Last_average_reward = r + gamma*Last_average_reward

                    """Plot the force and moments"""
                    # if render:
                    #     forcement.append(new_obs[0:6])
                    #     # print(forcement)
                    #     Force_moments.append(new_obs[0:6])
                    #     env.plot_force(forcement, t_rollout+1)

                    if epoch == 0 and cycle == 0:
                        forcement.append(new_obs[0:6])
                        Force_moments.append(new_obs[0:6])
                        # env.plot_force(forcement, t_rollout + 1)


                    if epoch == nb_epoch_cycles - 1 and cycle == nb_epoch_cycles - 1:
                        forcement.append(new_obs[0:6])
                        Force_moments.append(new_obs[0:6])
                        # env.plot_force(forcement, t_rollout + 1)

                    epoch_actions.append(action)
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs

                    """Episode done and start pull the pegs step by step"""
                    if done:
                        logger.info('Peg-in-hole assembly done!!!')
                        epoch_episode_rewards.append(episode_reward)
                        epoch_episodes_discount_reward.append(Last_average_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(t_rollout)
                        epoch_episodes += 1
                        # pull_done = False
                        # while pull_done is False and info:
                        #     pull_done, pull_safe = env.step_up() #Simulation env
                        #     pull_done, pull_safe = env.pull_up() #True env
                        #
                        # if pull_safe is False:
                        #     logger.info('Pull up the pegs failed for the exceed force!!!')
                        #     exit()
                        break

                    """Episode failed and start pull the pegs step by step"""
                    if info is False:
                        logger.info('Peg-in-hole assembly failed for the exceed force!!!')
                        # pull_done = False
                        # while pull_done is False and info:
                        #     pull_done, pull_safe = env.step_up()
                        #     pull_done, pull_safe = env.pull_up()  # True env
                        #
                        # if pull_safe is False:
                        #     logger.info('Peg-in-hole assembly failed for the exceed force!!!')
                        #     exit()

                        break

                Long_term_reward = Last_average_reward/Number_episodes
                epoch_qs.append(q_value)
                env.save_figure('force_moment')
                epoch_episodes_average_reward.append(Long_term_reward)
                agent.feedback_adptive_explore()
                if t_rollout == nb_rollout_steps - 1:
                    logger.info('Peg-in-hole assembly failed for exceed steps!!!')
                    logger.info('The deepest position'.format(obs[8]))

                """train model for nb_train_steps times"""
                for t_train in range(nb_train_steps):
                    cl, al = agent.train(epoch_actor_lr, epoch_critic_lr)
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

            """Adapt param noise, if necessary"""
            if memory.nb_entries >= batch_size and param_noise is not None:
                distance = agent.adapt_param_noise()
                epoch_adaptive_distances.append(distance)

            """write the result into the summary"""
            agent.log_scalar("actor_loss", mpi_mean(epoch_actor_losses), epoch_episodes)
            agent.log_scalar("critic_loss", mpi_mean(epoch_critic_losses), epoch_episodes)
            agent.log_scalar("episode_score", mpi_mean(epoch_episode_rewards), epoch_episodes)
            agent.log_scalar("episode_steps", mpi_mean(epoch_episode_steps), epoch_episodes)
            agent.log_scalar("episode_average_reward", mpi_mean(epoch_episodes_average_reward), epoch_episodes)
            agent.log_scalar("episode_discount_score", mpi_mean(epoch_episodes_discount_reward), epoch_episodes)

            """Log stats."""
            epoch_train_duration = time.time() - epoch_start_time
            stats = agent.get_stats()
            combined_stats = {}
            for key in sorted(stats.keys()):
                combined_stats[key] = mpi_mean(stats[key])

            """Rollout statistics. compute the mean of the total nb_epoch_cycles"""
            combined_stats['rollout/return'] = mpi_mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = mpi_mean(np.mean(episode_rewards_history))
            combined_stats['rollout/episode_steps'] = mpi_mean(epoch_episode_steps)
            combined_stats['rollout/episodes'] = mpi_sum(epoch_episodes)
            combined_stats['rollout/actions_mean'] = mpi_mean(epoch_actions)
            combined_stats['rollout/actions_std'] = mpi_std(epoch_actions)
            combined_stats['rollout/Q_mean'] = mpi_mean(epoch_qs)

            """Train statistics"""
            combined_stats['train/loss_actor'] = mpi_mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = mpi_mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = mpi_mean(epoch_adaptive_distances)

            """save the model and the result"""
            saver.save(sess, model_directory + 'simulation_model')
            # re_rewards = pd.DataFrame(epoch_episode_rewards)
            # re_rewards.to_csv("re_rewards.csv", sep=',', header=False, index=False)
            re_forcement = pd.DataFrame(Force_moments)
            re_forcement.to_csv(model_directory + 'simulation_forcement', sep=',', header=False, index=False)
            # re_steps = pd.DataFrame(epoch_episode_steps)
            # re_steps.to_csv("re_steps.csv", sep=',', header=False, index=False)
            # nf = pd.read_csv("data.csv", sep=',', header=None)

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])

            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)
