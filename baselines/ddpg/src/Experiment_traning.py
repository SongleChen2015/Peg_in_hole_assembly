# -*- coding: utf-8 -*-
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


def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    tau=0.001, eval_env=None, param_noise_adaption_interval=50, restore=False):

    rank = MPI.COMM_WORLD.Get_rank()
    max_action = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

    logger.info('scaling actions by {} before executing in env'.format(max_action))
    model_directory = '/home/rvsa/RL_project/Peg_in_Hole/1-baselines/baselines/ddpg/result/'

    agent = DDPG(actor, critic, memory, env.state_dim, env.action_dim,
                 gamma=gamma, tau=tau, normalize_returns=normalize_returns,
                 normalize_observations=normalize_observations,
                 batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
                 actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                 reward_scale=reward_scale)

    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    """Set up logging stuff only for a single worker"""
    saver = tf.train.Saver()


    # if rank == 0:
    #     saver = tf.train.Saver()
    # else:
    #     saver = None
    # eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)

    with U.single_threaded_session() as sess:
        """Prepare everything"""
        if restore:
            saver = tf.train.import_meta_graph(model_directory + 'model_fuzzy_new_3.meta')
            agent.restore_model(model_directory, saver, sess)
        else:
            agent.initialize(sess)
            sess.graph.finalize()

        """Agent Reset"""
        agent.reset()

        """Force calibration"""
        # env.robot_control.CalibFCforce()

        learning_epochs = 20
        delay_rate = np.power(10, 1/learning_epochs)

        """Revise the last epochs"""
        last_epochs = 0
        actor_lr = actor_lr/np.power(delay_rate, last_epochs)
        critic_lr = critic_lr/np.power(delay_rate, last_epochs)

        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        mean_rollout_time = []
        mean_epoch_rewards = []
        mean_epoch_steps = []
        mean_epoch_time = []
        epoch_adaptive_distances = []

        epoch_actions = []
        epoch_qs = []

        epoch_episodes = 0
        total_episodes = 0
        successful_rate = []
        Force_moments = np.zeros((1, 6))

        for epoch in range(nb_epochs):

            """Show the result for cycle 20 times and Save the model"""
            epoch_actor_losses = []
            epoch_critic_losses = []

            """Delay the learning rate"""
            epoch_actor_lr = actor_lr/delay_rate
            epoch_critic_lr = critic_lr/delay_rate
            epoch_start_time = time.time()

            for cycle in range(nb_epoch_cycles):

                """environment reset """
                agent.reset()
                obs = env.reset()
                episode_reward = 0.
                done = False
                rollout_start_time = time.time()
                force_array = np.zeros((150, 6))

                for t_rollout in range(nb_rollout_steps):

                    """Predict next action"""
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    assert action.shape[0] == env.action_dim

                    """scale for execution in env"""
                    new_obs, r, done, info = env.step(action, t_rollout)
                    # logger.info("The maximum force:" + str(max(abs(new_obs[0:3]))) + " The maximum moments:" +
                    #             str(max(abs(new_obs[3:6]))))
                    logger.info("The force:" + str(new_obs[0:3]) + " The moments:" + str(new_obs[3:6]))

                    episode_reward += r
                    force_array[t_rollout, :] = new_obs[0:6]
                    # Force_moments.append(new_obs[0:6])

                    """Plot the force and moments"""
                    if render:
                        env.plot_force(Force_moments, t_rollout)

                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done)

                    obs = new_obs

                    """Episode done and start pull the pegs step by step"""
                    if done:
                        logger.info('Peg-in-hole assembly done!!!')
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(t_rollout)
                        epoch_episodes += 1
                        pull_done = False
                        pull_safe = True
                        while pull_done is False and pull_safe:

                            pull_done, pull_safe = env.pull_up()  # True env

                        # if pull_safe is False:
                        #     logger.info('###############################################')
                        #     logger.info('Pull up the pegs failed for the exceed force!!!')
                        #     exit()
                        break

                    """Episode failed and start pull the pegs step by step"""
                    if info is False:
                        logger.info('Peg-in-hole assembly failed for the exceed force!!!')
                        pull_done = False
                        pull_safe = True
                        while pull_done is False and pull_safe:

                            pull_done, pull_safe = env.pull_up()  # True env

                        # if pull_safe is False:
                        #     logger.info('###############################################')
                        #     logger.info('Peg-in-hole assembly failed for the exceed force!!!')
                        #     exit()
                        break

                total_episodes += 1
                roluout_time = time.time() - rollout_start_time
                mean_rollout_time.append(roluout_time)
                Force_moments = np.concatenate((Force_moments, force_array), axis=0)

                if t_rollout == nb_rollout_steps-1:
                    logger.info('Peg-in-hole assembly failed for exceed steps!!!')
                    logger.info('The deepest position'.format(obs[8]))

                """train model for nb_train_steps times"""
                for t_train in range(nb_train_steps):
                    cl, al = agent.train(epoch_actor_lr, epoch_critic_lr)
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

            """Save the memory data"""
            agent.save_data()

            """Adapt param noise, if necessary"""
            if memory.nb_entries >= batch_size and param_noise is not None:
                distance = agent.adapt_param_noise()
                epoch_adaptive_distances.append(distance)

            # agent.feedback_adptive_explore()
            # agent.ou_adaptive_explore()
            """write the result into the summary"""
            agent.log_scalar("actor_loss", mpi_mean(epoch_actor_losses), epoch_episodes)
            agent.log_scalar("critic_loss", mpi_mean(epoch_critic_losses), epoch_episodes)
            agent.log_scalar("episode_score", mpi_mean(epoch_episode_rewards), epoch_episodes)
            agent.log_scalar("episode_steps", mpi_mean(epoch_episode_steps), epoch_episodes)

            """Log stats."""
            epoch_train_duration = time.time() - epoch_start_time
            mean_epoch_time.append(epoch_train_duration)

            """Successful rate"""
            successful_rate.append(epoch_episodes/total_episodes)
            stats = agent.get_stats()
            combined_stats = {}

            for key in sorted(stats.keys()):
                combined_stats[key] = mpi_mean(stats[key])

            """Rollout statistics. compute the mean of the total nb_epoch_cycles"""
            combined_stats['rollout/rewards'] = mpi_mean(epoch_episode_rewards)
            mean_epoch_rewards.append(mpi_mean(epoch_episode_rewards))
            combined_stats['rollout/episode_steps'] = mpi_mean(epoch_episode_steps)
            mean_epoch_steps.append(mpi_mean(epoch_episode_steps))
            # combined_stats['rollout/episodes'] = mpi_sum(epoch_episodes)
            combined_stats['rollout/actions_mean'] = mpi_mean(epoch_actions)
            combined_stats['rollout/actions_std'] = mpi_std(epoch_actions)
            combined_stats['rollout/Q_mean'] = mpi_mean(epoch_qs)

            """Train statistics"""
            combined_stats['train/loss_actor'] = mpi_mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = mpi_mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = mpi_mean(epoch_adaptive_distances)

            """Total statistics"""
            combined_stats['total/episodes'] = mpi_sum(epoch_episodes)
            combined_stats['total/epochs'] = epoch + 1

            """Plot reward and steps"""
            # env.plot_rewards(epoch_episode_rewards, epoch_episodes)
            # env.plot_steps(epoch_episode_steps, epoch_episodes)

            """save the model and the result"""
            saver.save(sess, model_directory + 'model_normal_new')

            pd_epoch_train_duration = pd.DataFrame(mean_epoch_time)
            pd_epoch_train_duration.to_csv('Experiment_data/epoch_train_duration_normal_new', sep=',', header=False, index=False)

            pd_rollout_time = pd.DataFrame(mean_rollout_time)
            pd_rollout_time.to_csv('Experiment_data/mean_rollout_time_normal_new', sep=',', header=False, index=False)

            pd_successful_rate = pd.DataFrame(successful_rate)
            pd_successful_rate.to_csv('Experiment_data/successful_rate_normal_new', sep=',', header=False, index=False)

            pd_Force_and_moments = pd.DataFrame(Force_moments)
            pd_Force_and_moments.to_csv("Experiment_data/force_moments_normal_new", sep=',', header=False, index=False)

            re_rewards = pd.DataFrame(epoch_episode_rewards)
            re_rewards.to_csv("Experiment_data/re_true_rewards_normal_new", sep=',', header=False, index=False)

            re_steps = pd.DataFrame(epoch_episode_steps)
            re_steps.to_csv("Experiment_data/re_true_steps_normal_new", sep=',', header=False, index=False)

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

            """Save the fig"""
            # env.save_figure('Result_figure')
