# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     Env_robot_control
   Description :  The class for real-world experiments to control the ABB robot,
                    which base on the basic class connect finally
   Author :       Zhimin Hou
   date：         18-1-9
-------------------------------------------------
   Change Activity:
                   18-1-9
-------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import Fuzzy_reward_mf_experiments as frf
import logging
import sys
sys.path.append("/home/rvsa/RL_project/Peg_in_Hole/1-baselines")
from baselines import logger, bench
from Connect_Finall import Robot_Control
import pandas as pd

logger.set_level(logging.DEBUG)
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

class Env_robot_control(object):
    def __init__(self):

        """state and Action Parameters"""
        self.action_dim = 6
        self.state_dim = 12
        self.terminal = False
        self.pull_terminal = False
        self.safe_else = True

        """control the bounds of the sensors"""
        # sensor_0~3 = f x, y, z
        # sensor_3~6 = m x, y, z
        # sensor 6~9 = x, y, z
        # sensor 9~12 = rx, ry, rz
        # self.sensors = [1./100, 1.0/100, 1.0/30, 1.0, 1.0, 1.0,
        #                        1.0/0.1, 1.0/0.1, 1.0/200., 1.0/180., 1.0/0.05, 1.0/0.05]
        # self.actions_bounds = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 2.])

        """Fx, Fy, Fz, Mx, My, Mz"""
        self.sensors = np.zeros(self.state_dim)
        self.init_state = np.zeros(self.state_dim)

        """dx, dy, dz, rx, ry, rz"""
        self.actions = np.zeros(self.action_dim)

        """good actions parameters"""
        self.Kpz = 0.015  # [0.01, 0.02]
        self.Krxyz = 0.01  # [0.001]
        self.Kpxy = 0.0022  # [0.0005, 0.002]
        self.Kdz = 0.002
        self.Kdxy = 0.0002
        self.Vel = 5.
        self.Kv = 0.5
        self.step_max = 65
        self.Kv_fast = 2.0934

        """The hole in world::::Tw_h=T*Tt_p, the matrix will change after installing again"""
        self.Tw_h = np.array([[0.9941, -0.1064, 0.0218, 531.3748],
                              [0.1064, 0.9943, -0.0032, -44.1109],
                              [-0.0213, 0.0055, 0.9998, 70.3943],
                              [0, 0, 0, 1.0000]])

        # self.Tw_h = np.array([[9.94127729e-01, -1.06342739e-01, 2.00328825e-02, 5.31604700e+02],
        #                       [1.06426394e-01, 9.94315580e-01, -3.16269280e-03, -4.43183873e+01],
        #                       [-1.95826778e-02, 5.26775395e-03, 9.99794319e-01, 7.06768898e+01],
        #                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # self.Tw_h = np.array([[ 9.93591091e-01,  -1.09760991e-01,   2.70058995e-02,   5.30701537e+02],
        #                       [ 1.09886144e-01,   9.93939061e-01,  -3.19878826e-03,  -4.43103953e+01],
        #                       [ -2.64911163e-02,   6.13746895e-03,   9.99630156e-01,   7.13090837e+01],
        #                       [ 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

        # self.Tw_h = np.array([[9.93966257e-01, -1.06298732e-01, 2.70500831e-02, 5.31283487e+02],
        #                       [1.06425708e-01, 9.94315198e-01, - 3.30302042e-03, - 4.41046845e+01],
        #                       [-2.65452019e-02, 6.15352237e-03, 9.99628623e-01, 7.11462325e+01],
        #                       [0.0, 0.0, 0.0, 1.0000]])

        # self.Tw_h = np.array([[9.94045497e-01, - 1.06350680e-01, 2.37302517e-02, 5.31279654e+02],
        #                       [1.06456211e-01, 9.94312159e-01, - 3.23400662e-03, - 4.43022530e+01],
        #                       [-2.32513390e-02, 5.73258896e-03, 9.99713167e-01, 7.14128785e+01],
        #                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        """The peg in tool, this is measured by the tracker"""
        self.Tt_p = np.array([[0.992324636460983, -0.123656366940134, -0.000957450195690770, 1.41144492938517],
                              [-0.123650357439767, -0.992313999324900, 0.00485764354118817, -0.848850965882200],
                              [-0.00155076978095379, -0.00469357455783674, -0.999987743210303, 129.363218736061],
                              [0, 0, 0, 1]])
        """The matrix of peg in tool, which is designed theoretically"""
        # self.Tt_p = np.array([[1, 0, 0, 0],
        #                       [0, -1, 0, 0],
        #                       [0, 0, -1, 130],
        #                       [0, 0, 0, 1]])

        """[Fx, Fy, Fz, Tx, Ty, Tz]"""
        self.refForce = [0, 0, -70, 0, 0, 0]
        self.refForce_pull = [0., 0., 80., 0., 0., 0.]

        """The safe force::F, M"""
        self.Safe_Force_Moment = [80, 5]

        """set up the figure and two subfigures"""
        self.fig = plt.figure('Experiments', figsize=(20, 20))
        self.ax_rewards = self.fig.add_subplot(221)
        self.ax_steps = self.fig.add_subplot(222)
        self.ax_force = self.fig.add_subplot(212)

        # self.render = render
        # plt.show(block=False)
        # plt.show()
        # plt.ion()

        """Build the controller and connect with robot"""
        self.robot_control = Robot_Control()

    """Motion step by step"""
    def step(self, action, step):

        """'Fuzzzy reward'; Including the steps and force; Only include the steps"""
        reward_methods = 'Fuzzy' #Time_force/Time
        """Get the model-basic action based on impendence control algorithm"""
        expert_actions = self.expert_actions(self.sensors[0:6])
        action = np.multiply(expert_actions, action + [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        """Get the current position"""
        Position, Euler, T = self.robot_control.GetCalibTool()

        """Velocity"""
        Vel = self.Kv_fast * sum(abs(self.sensors[0:6] - self.refForce))

        """Move and rotate the pegs"""
        self.robot_control.MoveToolTo(Position + action[0:3], Euler + action[3:6], Vel)

        """Get the next force"""
        self.sensors[0:6] = self.robot_control.GetFCForce()

        """Get the next position"""
        Position_next, Euler_next, T = self.robot_control.GetCalibTool()
        self.sensors[6:9] = Position_next
        self.sensors[9:12] = Euler_next

        """Whether the force&moment is safe for object"""
        max_abs_F_M = np.array([max(abs(self.sensors[0:3])), max(abs(self.sensors[3:6]))])
        self.safe_else = all(max_abs_F_M < self.Safe_Force_Moment)

        """Get the max force and moment"""
        f = max(abs(self.sensors[0:3]))
        m = max(abs(self.sensors[3:6]))
        z = self.sensors[8]
        max_depth = 40

        """reward for finished the task"""
        if z < 160:
            """change the reward"""
            Reward_final = 1.0 - step/self.step_max
            self.terminal = True
        else:
            Reward_final = 0.

        """Including three methods to design the reward function"""
        if reward_methods == 'Fuzzy':
            mfoutput, zdzoutput = frf.fuzzy_C1(m, f, 201 - z, action[5])
            Reward_process = frf.fuzzy_C2(mfoutput, zdzoutput)
        elif reward_methods == 'Time_force':
            force_reward = max(np.exp(0.02 * (f - 30)), np.exp(0.5 * (m - 1)))
            # force_reward = max(np.exp(0.01 * f), np.exp(0.3 * m)) #0.02, 0.5
            Reward_process = (-1) * (max_depth - (200 - z)) / max_depth * force_reward #[-1, 0]
        else:
            Reward_process = (-1) * (max_depth - (200 - z)) / max_depth

        Reward = Reward_final + Reward_process
        return self.sensors, Reward, self.terminal, self.safe_else

    """Pull the peg up by constant step"""
    def pull_up(self):
        """Only change the action_z"""
        action = np.array([0., 0., 2., 0., 0., 0.])

        """Get the current position"""
        Position, Euler, T = self.robot_control.GetCalibTool()

        """velocities"""
        Vel_up = self.Kv_fast * sum(abs(self.sensors[0:6] - self.refForce_pull))

        """move and rotate"""
        self.robot_control.MoveToolTo(Position + action[0:3], Euler, Vel_up)

        """Get the next force"""
        self.sensors[0:6] = self.robot_control.GetFCForce()

        """Get the next position"""
        Position_next, Euler_next, T = self.robot_control.GetCalibTool()
        self.sensors[6:9] = Position_next
        self.sensors[9:12] = Euler_next

        """if the force & moment is safe for object"""
        max_abs_F_M = [max(map(abs, self.sensors[0:3])), max(map(abs, self.sensors[3:6]))]
        self.safe_else = max_abs_F_M < self.Safe_Force_Moment

        """if finished"""
        z = self.sensors[8]
        if z > 202:
            self.pull_terminal = True

        return self.pull_terminal, self.safe_else

    """reset the start position or choose the fixed position move little step by step"""
    def reset(self):
        self.terminal = False
        self.pull_terminal = False
        self.safe_else = True
        self.Kp_z_0 = 0.93
        self.Kp_z_1 = 0.6

        Position_0, Euler_0, Twt_0 = self.robot_control.GetCalibTool()
        if Position_0[2] < 201:
            logger.info("The pegs didn't move the init position!!!")
            exit()

        """init_params: random"""
        # First_transformParas = np.array(
        #     [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1),
        #      np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), 3.0])

        """init_params: constant"""
        init_position = np.array([6.328895870000000e+02, -44.731415000000000, 3.497448430000000e+02])
        init_euler = np.array([1.798855440000000e+02, 1.306262000000000, -0.990207000000000])

        """Move to the target point quickly and align with the holes"""
        self.robot_control.MoveToolTo(init_position, init_euler, 20)
        self.robot_control.Align_PegHole()

        E_z = np.zeros(30)
        action = np.zeros((30, 3))
        """Move by a little step"""
        for i in range(30):

            myForceVector = self.robot_control.GetFCForce()
            if max(abs(myForceVector[0:3])) > 5:
                logger.info("The pegs can't move for the exceed force!!!")
                exit()

            """"""
            Position, Euler, Tw_t = self.robot_control.GetCalibTool()
            print(Position)

            Tw_p = np.dot(Tw_t, self.robot_control.Tt_p)
            print(self.robot_control.Tw_h[2, 3])

            E_z[i] = self.Tw_h[2, 3] - Tw_p[2, 3]
            print(E_z[i])

            if i < 3:
                action[i, :] = np.array([0., 0., self.Kp_z_0*E_z[i]])
                vel_low = self.Kv * abs(E_z[i])
            else:
                # action[i, :] = np.array([0., 0., action[i-1, 2] + self.Kp_z_0*(E_z[i] - E_z[i-1])])
                action[i, :] = np.array([0., 0., self.Kp_z_1*E_z[i]])
                vel_low = min(self.Kv * abs(E_z[i]), 0.5)


            self.robot_control.MoveToolTo(Position + action[i, :], Euler, vel_low)
            print(action[i, :])

            if abs(E_z[i]) < 0.001:
                logger.info("The pegs reset successfully!!!")
                self.init_state[0:6] = myForceVector
                self.init_state[6:9] = Position
                self.init_state[9:12] = Euler
                break

        return self.init_state

    # """Calibrate the force sensor"""
    # def CalibFCforce(self):
    #     if self.robot_control().CalibFCforce():
    #         init_Force = self.robot_control.GetFCForce()
    #         if max(abs(init_Force)) > 1:
    #             logger.info("The Calibration of Force Failed!!!")
    #             exit()
    #     logger.info("The Calibration of Force Finished!!!")
    #     return True

    """Get the states and limit it the range"""
    def get_state(self):
        s = self.sensors.astype(np.float32)
        return s

    """use to show the simulation and plot the model"""
    # def plot_model(self, show_des):
    #     if show_des is True:
    #         self.PegInHolemodel.plot(self.ax_model, self.PegInHolemodel.Centers_pegs,
    #                                  self.PegInHolemodel.Centers_holes,
    #                                  self.PegInHolemodel.pegs, self.PegInHolemodel.holes)
    #         plt.pause(0.01)
    #         # while True:
    #         #     plt.pause(0.1)

    """Get the fuzzy control actions"""
    def expert_actions(self, state):
        """PID Controller"""
        action = np.zeros(6)

        """The direction of Mx same with Rotx; But another is oppsite"""
        Force_error = self.refForce - state

        Force_error[0] = (-1) * Force_error[0]

        """rotate around X axis"""
        action[3] = (-1)*self.Krxyz * Force_error[3]

        """rotate around Y axis and Z axis"""
        action[4:6] = self.Krxyz * Force_error[4:6]

        """move along the X and Y axis"""
        action[0:2] = self.Kpxy * Force_error[0:2]

        """move along the Z axis"""
        action[2] = self.Kpz * Force_error[2]
        return action

    """Plot the six forces"""
    def plot_force(self, Fores_moments, steps):

        # fig_force = plt.figure("Simulation")
        # plt.figure("ForceAndMoment")
        # plt.clf()
        # plt.getp(self.fig_forcemoments)
        plt.ion()
        # force_moment = np.array(Fores_moments).transpose()
        force_moment = np.array(Fores_moments)
        steps_lis = np.linspace(0, steps - 1, steps)
        ax_force = self.ax_force
        ax_force.clear()
        ax_force.plot(steps_lis, force_moment[0, :], c=COLORS[0], linewidth=2.5, label="Force_X")
        ax_force.plot(steps_lis, force_moment[1, :], c=COLORS[1], linewidth=2.5, label="Force_Y")
        ax_force.plot(steps_lis, force_moment[2, :], c=COLORS[2], linewidth=2.5, label="Force_Z")
        ax_force.plot(steps_lis, 10 * force_moment[3, :], c=COLORS[3], linewidth=2.5, label="Moment_X")
        ax_force.plot(steps_lis, 10 * force_moment[4, :], c=COLORS[4], linewidth=2.5, label="Moment_Y")
        ax_force.plot(steps_lis, 10 * force_moment[5, :], c=COLORS[5], linewidth=2.5, label="Moment_Z")

        ax_force.set_xlim(0, 40)
        ax_force.set_xlabel("Steps")
        ax_force.set_ylim(-50, 50)
        ax_force.set_ylabel("Force(N)/Moment(Ndm)")
        ax_force.legend(loc="upper right")
        ax_force.grid()
        # self.fig_forcemoments.savefig("Force_moment.jpg")
        # plt.show()
        plt.show(block=False)
        plt.pause(0.1)


    def plot_rewards(self, Rewards, steps):

        # rewards = np.array(Rewards).transpose()
        rewards = np.array(Rewards)
        steps_lis = np.linspace(0, steps - 1, steps)

        self.ax_rewards.clear()
        self.ax_rewards.plot(steps_lis, rewards, c=COLORS[0], label='Rewards_Episode')

        self.ax_rewards.set_xlim(0, 1000)
        self.ax_rewards.set_xlabel("Episodes")
        self.ax_rewards.set_ylim(-50, 50)
        self.ax_rewards.set_ylabel("Reward")
        self.ax_rewards.legend(loc="upper right")
        self.ax_rewards.grid()

        plt.show(block=False)
        plt.pause(0.1)


    def plot_steps(self, episode_steps, steps):

        # Episode_steps = np.array(episode_steps).transpose()
        Episode_steps = np.array(episode_steps)
        steps_lis = np.linspace(0, steps - 1, steps)

        self.ax_steps.plot(steps_lis, Episode_steps, c=COLORS[1], label='Steps_Episode')

        self.ax_steps.set_xlim(0, 1000)
        self.ax_steps.set_xlabel("Episodes")
        self.ax_steps.set_ylim(0, 60)
        self.ax_steps.set_ylabel("steps")
        self.ax_steps.legend(loc="upper right")
        self.ax_steps.grid()
        plt.show(block=False)
        plt.pause(0.1)

    """Save the figure"""
    def save_figure(self, figname):

        self.fig.savefig(figname + '.jpg')

    """Read the csv"""
    def Plot_from_csv(self, csvname, figname):
        nf = pd.read_csv(csvname + ".csv", sep=',', header=None)

        fig = plt.figure(figname, figsize=(8, 8))
        ax_plot = fig.add_subplot()

        steps = np.linspace(0, len(nf) - 1, len(nf))
        plt.plot(steps, nf, c=COLORS[0], linewidth=2.5)

        ax_plot.set_xlim(0, 40)
        ax_plot.set_xlabel("Steps")
        ax_plot.set_ylim(-50, 50)
        ax_plot.set_ylabel("Force(N)/Moment(Ndm)")
        ax_plot.legend(loc="upper right")
        ax_plot.grid()