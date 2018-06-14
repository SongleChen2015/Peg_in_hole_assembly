# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     Plot_result
   Description :
   Author :       Zhimin Hou
   date：         18-1-15
-------------------------------------------------
   Change Activity:
                   18-1-15
-------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']


"""Read the csv"""
def Plot_average_from_csv(csvname, xlabel, ylabel):
    nf = pd.read_csv(csvname, sep=',', header=None)

    average_nf = np.zeros(15)
    for i in range(15):
        average_nf[i] = sum(nf[0][i * 10:(10 * i + 10)]) / 10

    fig = plt.figure(figsize=(12, 8))
    ax_plot = fig.add_subplot(111)

    steps = np.linspace(1, len(average_nf), len(average_nf))
    plt.plot(steps, list(average_nf), c=COLORS[2], linewidth=2.5, label=ylabel)

    ax_plot.set_xlim(0, 16)
    ax_plot.set_xlabel(xlabel, fontsize=32)
    ax_plot.set_ylabel(ylabel, fontsize=32)
    # ax_plot.set_ylim(130, 50)

    plt.xticks([0, 2., 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
               [1, 20, 40, 60, 80, 100, 120, 140, 160], fontsize=26)
    plt.yticks(fontsize=26)

    # ax_plot.legend(loc="upper right")
    ax_plot.grid()
    fig.savefig('/home/rvsa/RL_project/result_experiment/' + ylabel + '_normal.eps')
    fig.savefig('/home/rvsa/RL_project/result_experiment/' + ylabel + '_normal.png')
    plt.show()

def Plot_more_crves_average_from_csv(csvname1, csvname2, xlabel, ylabel):
    nf1 = pd.read_csv(csvname1, sep=',', header=None)
    nf2 = pd.read_csv(csvname2, sep=',', header=None)

    number_episode = 15
    average_nf1 = np.zeros(number_episode)
    average_nf2 = np.zeros(number_episode)

    for i in range(number_episode):
        average_nf1[i] = sum(nf1[0][i * 10:(10 * i + 10)]) / 10
        average_nf2[i] = sum(nf2[0][i * 10:(10 * i + 10)]) / 10

    fig = plt.figure(figsize=(12, 8))
    ax_plot = fig.add_subplot(111)

    steps = np.linspace(1, len(average_nf1), len(average_nf1))
    plt.plot(steps, list(average_nf1), c=COLORS[2], linewidth=2.5, label=csvname1)
    plt.plot(steps, list(average_nf2), c=COLORS[3], linewidth=2.5, label=csvname2)

    ax_plot.set_xlim(0, 16)
    ax_plot.set_xlabel(xlabel, fontsize=32)
    ax_plot.set_ylabel(ylabel, fontsize=32)
    # ax_plot.set_ylim(130, 50)

    plt.xticks([0, 2., 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
               [1, 20, 40, 60, 80, 100, 120, 140, 160], fontsize=26)
    plt.yticks(fontsize=26)

    ax_plot.legend(loc="upper right")
    ax_plot.grid()
    # fig.savefig('/home/rvsa/RL_project/result_experiment/' + ylabel + '_compare.eps')
    # fig.savefig('/home/rvsa/RL_project/result_experiment/' + ylabel + '_compare.png')
    plt.show()

def Plot_from_csv(csvname, xlabel, ylabel):
    nf = pd.read_csv(csvname, sep=',', header=None)

    fig = plt.figure(figsize=(12, 8))
    ax_plot = fig.add_subplot(111)

    steps = np.linspace(1, len(nf), len(nf))
    # steps_10 = np.linspace(0, 10 * (len(nf) - 1), len(nf))
    plt.plot(steps, nf, c=COLORS[2], linewidth=2.5, label=ylabel)

    ax_plot.set_xlim(0, 16)
    ax_plot.set_xlabel(xlabel, fontsize=32)

    # ax_plot.set_ylim(1600, 1700)
    ax_plot.set_ylabel(ylabel, fontsize=32)
    # print(steps)
    # print(steps_10)

    plt.xticks([0, 2., 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
               [1, 20, 40, 60, 80, 100, 120, 140, 160], fontsize=26)
    plt.yticks(fontsize=26)
    # ax_plot.set_xticks([20.0], [200.])
    # ax_plot.legend(loc="upper right")
    ax_plot.grid()
    fig.savefig('/home/rvsa/RL_project/result_experiment/' + ylabel + '_normal.eps')
    plt.show()

def Plot_more_crves_from_csv(csvname1, csvname2, xlabel, ylabel):
    nf1 = pd.read_csv(csvname1, sep=',', header=None)
    nf2 = pd.read_csv(csvname2, sep=',', header=None)

    nf1 = nf1[0:8]
    nf2 = nf2[0:8]
    fig = plt.figure(figsize=(12, 8))
    ax_plot = fig.add_subplot(111)

    steps = np.linspace(1, len(nf1), len(nf1))
    # steps_10 = np.linspace(0, 10 * (len(nf) - 1), len(nf))
    plt.plot(steps, nf1, c=COLORS[2], linewidth=2.5, label=csvname1)
    plt.plot(steps, nf2, c=COLORS[3], linewidth=2.5, label=csvname2)

    ax_plot.set_xlim(0, 16)
    ax_plot.set_xlabel(xlabel, fontsize=32)

    # ax_plot.set_ylim(1600, 1700)
    ax_plot.set_ylabel(ylabel, fontsize=32)
    # print(steps)
    # print(steps_10)

    plt.xticks([0, 2., 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
               [1, 20, 40, 60, 80, 100, 120, 140, 160], fontsize=26)
    plt.yticks(fontsize=26)
    # ax_plot.set_xticks([20.0], [200.])
    # ax_plot.legend(loc="upper right")
    ax_plot.grid()
    # fig.savefig('/home/rvsa/RL_project/result_experiment/' + ylabel + '_compare.eps')
    plt.show()

def Plot_force_moment(csvname):
    nf = pd.read_csv(csvname, sep=',', header=None)

    fig = plt.figure(figsize=(12, 10))
    ax_plot = fig.add_subplot(111)


    # plt.ion()
    steps = 52
    # force_moment = np.array(Fores_moments).transpose()
    # force_moment = np.array(Fores_moments)
    # print(np.array(nf[1:20]))
    force_moment = np.array(nf[1:53])
    steps_lis = np.linspace(0, steps - 1, steps)
    ax_force = ax_plot
    ax_force.clear()
    ax_force.plot(steps_lis, force_moment[:, 0], c=COLORS[0], linewidth=2.5, label="Force_X")
    ax_force.plot(steps_lis, force_moment[:, 1], c=COLORS[1], linewidth=2.5, label="Force_Y")
    ax_force.plot(steps_lis, force_moment[:, 2], c=COLORS[2], linewidth=2.5, label="Force_Z")
    ax_force.plot(steps_lis, 10 * force_moment[:, 3], c=COLORS[3], linewidth=2.5, label="Moment_X")
    ax_force.plot(steps_lis, 10 * force_moment[:, 4], c=COLORS[4], linewidth=2.5, label="Moment_Y")
    ax_force.plot(steps_lis, 10 * force_moment[:, 5], c=COLORS[5], linewidth=2.5, label="Moment_Z")

    ax_force.set_xlim(0, 60)
    ax_force.set_xlabel("Steps", fontsize=18)
    ax_force.set_ylim(-100, 100)
    ax_force.set_ylabel("Force(N)/Moment(Nm x 10)", fontsize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    ax_force.legend(loc="upper right", fontsize=16)
    ax_force.grid()
    # self.fig_forcemoments.savefig("Force_moment.jpg")
    # plt.show()
    plt.show(block=True)
    # plt.pause(0.1)

# Plot_force_moment('force_moments_normal')

# Plot_average_from_csv('total_steps_add_force', 'Episodes', 'Episode_Steps')
# Plot_average_from_csv('total_rewards_add_force', 'Episodes', "Episode_Reward")

# Plot_from_csv('epoch_duration_fuzzy_reward', "Episodes", 'Episode_time')
# Plot_from_csv('epoch_train_duration_general', "Episodes", 'Episode_time')
# Plot_from_csv('total_normal_duration', "Episodes", 'Episode_time')

Plot_more_crves_average_from_csv('total_steps_add_force', 'total_rewards_add_force', 'Episodes', "Episode_Steps")
# Plot_more_crves_from_csv('epoch_duration_fuzzy_reward', 'epoch_train_duration_general', "Episodes", 'Episode_time')


"""Contact the two data"""
# nf_time1 = pd.read_csv('re_true_steps_add_force', sep=',', header=None)
# nf_time2 = pd.read_csv('re_true_steps_add_force_1', sep=',', header=None)
#
# nf_time3 = pd.concat([nf_time1, nf_time2])
# nf_time3.to_csv('total_steps_add_force', sep=',', header=False, index=False)




# nf_1 = pd.read_csv('re_true_rewards.csv', sep=',', header=None)
# nf_2 = pd.read_csv('re_true_rewards_1', sep=',', header=None)
# nf_3 = pd.concat([nf_1, nf_2])
# # nf_3 = pd.merge(nf_1, nf_2, how='left')
# nf_3.to_csv('total_rewards', sep=',', header=False, index=False)
#
# average_nf = np.zeros(10)
# for i in range(10):
#     average_nf[i] = sum(nf[0][i * 10:(10 * i + 10)]) / 10
# average_nf = np.zeros(10)
# for i in range(9):
#     average_nf[i] = sum(nf[0][i*10:(10 * i + 10)]) / 10
# print(average_nf)
# actor_lr = 1e-3
# learning_epochs = 100
#
# delay_rate = np.power(10, 1/learning_epochs)
#
# """Revise the last epochs"""
# last_epochs = 10
# actor_lr = actor_lr/np.power(delay_rate, last_epochs)
#
# print(actor_lr)