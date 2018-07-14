import sys
sys.path.append("/home/rvsa/RL_project/Peg_in_Hole/1-baselines")
import numpy as np

# mfoutput_1, zdzoutput_1 = f.fuzzy_C1(0.6, 10, 40, 0.6)
# Reward = f.fuzzy_C2(mfoutput_1, zdzoutput_1)
# print(Reward-1)
#
# mfoutput, zdzoutput = frf.fuzzy_C1(1.6, 20, 10, 1.6)
# Reward_process = frf.fuzzy_C2(mfoutput, zdzoutput)
# print(Reward_process)

print(np.exp(0.05))
# print(np.power(10, 0))
# reward = []
# reward.append(2)
# reward.append(3)
# a = []
# steps = np.zeros((10, 5))
# a.append(steps)
# d = np.zeros((5, 5))
# a.append(d)
# print(list(a))
# # # steps.append(1.)
# # # steps.append(2.)
# # df = pd.DataFrame(reward)
# dn = pd.DataFrame(a)
# # print(list(steps))
# # print(steps)
# dn.to_csv("test.csv", sep=',', header=False, index=False)
# dn.to_csv("data.csv", sep=',', header=False, index=False)
#
#
# print(max(np.array([0, 0, 50])))
# env = Env_robot_control()



# def Plot_from_csv(csvname, figname):
#     nf = pd.read_csv(csvname + ".csv", sep=',', header=None)
#
#     fig = plt.figure(figname, figsize=(8, 8))
#     ax_plot = fig.add_subplot(111)
#
#     steps = np.linspace(0, len(nf) - 1, len(nf))
#     plt.plot(steps, nf, c='b', linewidth=2.5, label='Reward_episode')
#
#     ax_plot.set_xlim(0, 40)
#     ax_plot.set_xlabel("Steps")
#     ax_plot.set_ylim(-50, 50)
#     ax_plot.set_ylabel("Force(N)/Moment(Ndm)")
#     ax_plot.legend(loc="upper right")
#     ax_plot.grid()
#     plt.show()
#
# Plot_from_csv('re_rewards', 'Reward')


# a = []
# for i in range(10):
#     a.append(i+1)
#
# nf = pd.read_csv("re_steps.csv", sep=',', header=None)
#
# steps = np.linspace(0, 9, 10)
# plt.plot(steps, np.array(a))
# plt.show()


# env.plot_rewards(nf[0], len(nf[0]))

# Tw_h = np.array([[0.9941, -0.1064, 0.0218, 531.3748],
#                               [0.1064, 0.9943, -0.0032, -44.1109],
#                               [-0.0213, 0.0055, 0.9998, 70.3943],
#                               [0, 0, 0, 1.0000]])
#
# def MatrixToEuler(T):
#     Position = np.zeros(3, dtype=float)
#     Position[0] = T[0, 3]
#     Position[1] = T[1, 3]
#     Position[2] = T[2, 3]
#
#     Euler = np.zeros(3, dtype=float)
#     Euler[2] = np.arctan2(T[1, 0], T[0, 0])
#     Euler[1] = np.arctan2(-T[2, 0], np.cos(Euler[2]) * T[0, 0] + np.sin(Euler[2]) * T[1, 0])
#     Euler[0] = np.arctan2(np.sin(Euler[2]) * T[0, 2] - np.cos(Euler[2]) * T[1, 2],
#                           -np.sin(Euler[2]) * T[0, 1] + np.cos(Euler[2]) * T[1, 1])
#
#     Euler = Euler * 180 / np.pi
#
#     return Position, Euler
#
# T = np.array([[1, 1], [0.5, 2]])
# S = np.array([[2, 3], [0.3, 4]])
# print(np.linalg.inv(T))
#
# Position, Euler = MatrixToEuler(Tw_h)
# print(Position)
# print(Euler)
# env = gym.make('Pendulum-v0')
#
# observation = env.reset()
# env.render()


# a = np.array([1, float("nan"), float("-inf")])
# b = np.isinf(a)
# c = np.isnan(a)


# try:
#     print(np.isinf(d))

# except:
#     print("dsdsds")

# assert np.isnan(a).all() == False
    # print(np.isinf(a))