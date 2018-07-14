import numpy as np
import matplotlib.pyplot as plt
from Simulation_PegHole import DualPegsinHoles
# import fuzzy_reward_function as frf
# import New_fuzzy_reward_mf_important as frf
import Fuzzy_reward_mf_simulation as frf
import logging
import sys
sys.path.append("/home/rvsa/RL_project/Peg_in_Hole/1-baselines")
from baselines import logger, bench

logger.set_level(logging.DEBUG)
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

class PegintoHoles(object):
    def __init__(self, render=False):
        self.action_dim = 6
        self.state_dim = 12

        self.terminal = False # if finished
        self.pull_terminal = False
        self.PegInHolemodel = None
        self.safe_else = True
        # control the bounds of the sensors
        # sensor_0~3 = f x, y, z
        # sensor_3~6 = m x, y, z
        # sensor 6~9 = x, y, z
        # sensor 9~12 = rx, ry, rz
        # self.sensors = [1./100, 1.0/100, 1.0/30, 1.0, 1.0, 1.0,
        #                        1.0/0.1, 1.0/0.1, 1.0/200., 1.0/180., 1.0/0.05, 1.0/0.05]
        # self.actions_bounds = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 2.])
        self.sensors = np.zeros(self.state_dim)
        self.actions = np.zeros(self.action_dim)

        # good actions parameters
        self.K_reward = -0.05 #0.2# self.Kv = 2
        self.Kpz = 0.015 # [0.01, 0.02]
        self.Krxyz = 0.001 # [0.001]
        self.Kpxy = 0.0008 # [0.0005, 0.002]
        self.step_max = 50

        # self.Kdz = 0.002
        # self.Kdxy = 0.0002
        self.refForce = [0, 0, -100, 0, 0, 0]
        self.Safe_Force_Moment = [100, 30]

        """set up the figure and two subfigures"""
        # self.fig = plt.figure('Simulation', figsize=(20, 10))
        # self.ax_model = self.fig.add_subplot(121, projection='3d')
        # self.ax_force = self.fig.add_subplot(122)
        self.foce_fig = plt.figure('Simulation_force', figsize=(10, 7.5))
        self.ax_force = self.foce_fig.add_subplot(111)
        self.render = render

    # input action output next_state and reward
    # accroding to the parameters Kx, Kz, Ky, Ka, Kb, Kv to adjust the forces
    def step(self, action, step):

        # get the next state
        expert_actions = self.expert_actions(self.sensors[0:6])
        # based on the expert actions
        action = np.multiply(expert_actions, action + [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        NewsumContactForceInPeg, NewPosition = self.PegInHolemodel.ChangeOneStep(action)
        self.sensors[0:6] = NewsumContactForceInPeg
        self.sensors[6:12] = NewPosition

        # if the force & moment is safe for object
        max_abs_F_M = [max(map(abs, self.sensors[0:3])), max(map(abs, self.sensors[3:6]))]
        self.safe_else = max_abs_F_M < self.Safe_Force_Moment

        """ fuzzy reward function """
        # f = max(abs(self.sensors[0:3]))
        # m = max(abs(self.sensors[3:6]))
        f = max(abs(self.sensors[0:3]))
        m = max(abs(self.sensors[3:6]))
        z = self.sensors[8]

        # if it's finished
        if z < 150:
            """change the reward"""
            Reward_final = 1.0 - step/self.step_max
            self.terminal = True
        else:
            Reward_final = 0.

        mfoutput, zdzoutput = frf.fuzzy_C1(m, f, 200 - z, action[5])
        Reward_process = frf.fuzzy_C2(mfoutput, zdzoutput)

        # Reward_final = 0.
        max_depth = 50
        """The normal reward"""
        # Reward_process = (-1) * (max_depth - (200 - z))/max_depth

        # force_reward = max(np.exp(0.01 * f), np.exp(0.1 * m))
        # force_reward = max(np.exp(0.03 * (f - 20)), np.exp(m - 2))
        # force_reward = max(np.exp(-0.6 * (f)), np.exp(-0.5 * m))
        # force_reward = max(np.exp(-0.03 * (f)), np.exp(-0.5 * m))

        """Including the steps and force"""
        # if step < 20:
        #     # force_reward = max(np.exp(f - 10), np.exp(m - 1))
        #     Reward_process = (-1) * (max_depth - (200 - z))/max_depth
        # else:
        #     Reward_process = (-1) * max(np.exp(0.03 * (f - 20)), np.exp(m - 0.5))

        # Reward_process = (-1) * max(np.exp(0.03 * (f - 20)), np.exp(m - 0.5))
        # Reward_process = (-1) * (max_depth - (200 - z))/max_depth * force_reward

        Reward = Reward_final + Reward_process

        return self.get_state(), Reward, self.terminal, self.safe_else, expert_actions

    """Pull the peg up"""
    def step_up(self):
        action = np.array([0., 0., 0., 0., 0., -2.0])
        NewsumContactForceInPeg, NewPosition = self.PegInHolemodel.ChangeOneStep(action)
        self.sensors[0:6] = NewsumContactForceInPeg
        self.sensors[6:12] = NewPosition
        max_abs_F_M = [max(map(abs, self.sensors[0:3])), max(map(abs, self.sensors[3:6]))]
        self.safe_else = max_abs_F_M < self.Safe_Force_Moment

        """if the pull finished"""
        if self.sensors[8] > 200:
            self.pull_terminal = True

        return self.pull_terminal, self.safe_else

    # create the simulation
    def reset(self):
        self.terminal = False
        self.pull_terminal = False
        self.safe_else = True
        """init_params: random"""
        First_transformParas = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1),
                                         np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), 3.0])

        """init_params: constant"""
        # First_transformParas = np.array([0., 0.08, -0.09, 0.01, -0.1, 3])

        logger.info("Model the init state:", str(First_transformParas))
        self.PegInHolemodel = DualPegsinHoles(First_transformParas)
        self.sensors[0:6] = self.PegInHolemodel.sumContactForceInPeg
        self.sensors[6:12] = self.PegInHolemodel.Position
        return self.get_state()

    # limit the range of states
    def get_state(self):
        s = self.sensors.astype(np.float32)
        return s

    # use the actions of fuzzy control method:: get the different parameters of controller
    def good_actions(self):
        action = np.zeros(3)
        errorForce = self.sensors[0:6] - self.refForce
        # # action[3:5] = self.Kpxy*errorForce[0:2]
        # action = self.Kpz*errorForce[2]
        # three rotate actions
        action[0:3] = self.Kpz * errorForce[3:6]
        return action

    # use to show the simulation
    def plot_model(self, show_des):
        if show_des is True:
            self.PegInHolemodel.plot(self.ax_model, self.PegInHolemodel.Centers_pegs, self.PegInHolemodel.Centers_holes,
                                     self.PegInHolemodel.pegs, self.PegInHolemodel.holes)
            plt.pause(0.01)
            # while True:
            #     plt.pause(0.1)

    def expert_actions(self, state):

        action = np.zeros(6)
        Force_error = state - self.refForce
        # rotate around three axis
        action[0:3] = self.Krxyz * Force_error[3:6]
        # move along the X and Y axis
        action[3:5] = self.Kpxy * Force_error[0:2]
        # move along the Z axis
        action[5] = self.Kpz * Force_error[2]

        # Force_error = self.refForce - state
        # Force_error[0] = (-1) * Force_error[0]
        #
        # """rotate around X axis"""
        # action[0] = (-1) * self.Krxyz * Force_error[3]
        #
        # """rotate around Y axis and Z axis"""
        # action[1:3] = self.Krxyz * Force_error[4:6]
        #
        # """move along the X and Y axis"""
        # action[3:5] = self.Kpxy * Force_error[0:2]
        #
        # """move along the Z axis"""
        # action[5] = self.Kpz * Force_error[2]

        return action

    def plot_force(self, force_moment, steps):
        # fig_force = plt.figure("Simulation")
        # plt.figure("ForceAndMoment")
        # plt.clf()
        # plt.getp(self.fig_forcemoments)
        plt.ion()
        # force_moment = np.array(force_moment).transpose()
        force_moment = np.array(force_moment)
        steps = np.linspace(0, steps-1, steps)
        ax_force = self.ax_force
        ax_force.clear()
        ax_force.plot(steps, force_moment[:, 0], c=COLORS[0], linewidth=2.5, label="Force_X")
        ax_force.plot(steps, force_moment[:, 1], c=COLORS[1], linewidth=2.5, label="Force_Y")
        ax_force.plot(steps, force_moment[:, 2], c=COLORS[2], linewidth=2.5, label="Force_Z")
        ax_force.plot(steps, 10*force_moment[:, 3], c=COLORS[3], linewidth=2.5, label="Moment_X")
        ax_force.plot(steps, 10*force_moment[:, 4], c=COLORS[4], linewidth=2.5, label="Moment_Y")
        ax_force.plot(steps, 10*force_moment[:, 5], c=COLORS[5], linewidth=2.5, label="Moment_Z")

        ax_force.set_xlim(0, 40)
        ax_force.set_xlabel("Steps", fontsize=18)
        ax_force.set_ylim(-20, 20)
        ax_force.set_ylabel("Force(N)/Moment(Ndm)", fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax_force.legend(loc="upper right")
        ax_force.grid()

        plt.pause(0.1)
        plt.show(block=False)
        # self.fig_forcemoments.savefig("Force_moment.jpg")
        # plt.show()

    def plot_figure(self, force_moment, steps, savefig):
        # self.fig = plt.figure('Simulation')
        # self.ax_model = self.fig.add_subplot(121, projection='3d')
        # self.ax_force = self.fig.add_subplot(122)
        if self.render:
            plt.show()
            # plt.ion()
        self.PegInHolemodel.plot(self.ax_model, self.PegInHolemodel.Centers_pegs, self.PegInHolemodel.Centers_holes,
                                 self.PegInHolemodel.pegs, self.PegInHolemodel.holes)

        self.plot_force(force_moment, steps)
        if savefig:
            self.fig.savefig("Simulation.jpg")

    """Save the figure"""

    def save_figure(self, figname):

        self.foce_fig.savefig(figname + '.eps')





# test
# Env_model = PegintoHoles()
# sensors = Env_model.reset()
# print(sensors[8])

# stop = False
# while stop is False:
#     s, r, stop, e = Env_model.step([0., 0., 0., 0., 0., 2.])
#     Env_model.plot_model(True)



# First_transformParas = np.array([np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), 0.0,
#                                  np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), 2.0])
# First_transformParas = np.array([0.05, 0., 0., 0., 0., 2.])
# model = OnePeginHole(First_transformParas)
# # diameterHole = 30
# # diameterPeg = 29.9
# # Center = [0, 0, 0]
# # height = 100
# # Centers_pegs, Centers_holes, pegs, holes = model.createPegInHoleModel(diameterHole, diameterPeg, Center, height)
