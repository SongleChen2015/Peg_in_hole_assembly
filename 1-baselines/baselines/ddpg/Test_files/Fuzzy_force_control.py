import matplotlib.pyplot as plt
import numpy as np

from Test_files.EnvPeginHoles import PegintoHoles

"""
make use of the traditional forcr control method 
"""
"""Set parameters"""
Kpz = 0.015
Krxyz = 0.001
Kpxy = 0.0008
Kdz = 0.002
Kdxy = 0.0002
Kv = 2
RefForce = [0, 0, -100, 0, 0, 0]
Env_model = PegintoHoles(True)
Env_model.reset()
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

def get_actions(state):
    action = np.zeros(6)
    Force_error = state - RefForce
    # rotate around three axis
    action[0:3] = Krxyz * Force_error[3:6]
    # move along the X and Y axis
    action[3:5] = Kpxy * Force_error[0:2]
    # move along the Z axis
    action[5] = Kpz * Force_error[2]
    return action

name = 'force'
def plot_force(force_moment, steps):

    ax_force = fig.add_subplot(111)
    # plt.figure("ForceAndMoment")
    # plt.clf()
    # plt.getp(fig)
    # plt.ion()
    ax_force.clear()
    ax_force.plot(steps, force_moment[0, :], c=COLORS[0], linewidth=2.5, label="Force_X")
    ax_force.plot(steps, force_moment[1, :], c=COLORS[1], linewidth=2.5, label="Force_Y")
    ax_force.plot(steps, force_moment[2, :], c=COLORS[2], linewidth=2.5, label="Force_Z")
    ax_force.plot(steps, 10*force_moment[3, :], c=COLORS[3], linewidth=2.5, label="Moment_X")
    ax_force.plot(steps, 10*force_moment[4, :], c=COLORS[4], linewidth=2.5, label="Moment_Y")
    ax_force.plot(steps, 10*force_moment[5, :], c=COLORS[5], linewidth=2.5, label="Moment_Z")

    ax_force.set_xlim(0, 40)
    ax_force.set_xlabel("Steps")
    ax_force.set_ylim(-50, 50)
    ax_force.set_ylabel("Force(N)/Moment(Ndm)")
    ax_force.legend(loc="upper right")
    ax_force.grid()

    plt.show(block=False)
    # plt.pause(0.0001)
    fig.savefig(name + '.jpg')

steps = 0
finish = False
Fores_moments = []
espoide_reward = 0.
plt.clf()
fig = plt.figure('Simulation', figsize=(20, 20))
while finish is False:
    S_current = Env_model.get_state()
    print(S_current[8])
    force_moment = S_current[0:6]
    print(force_moment)
    actions = get_actions(S_current[0:6])
    S_next, reward, finish, safe_else = Env_model.step(actions, steps+1)
    espoide_reward +=reward
    # Env_model.plot_model(True)
    if S_current[8]<100:
        break
    steps += 1
    steps_lis = np.linspace(0, steps - 1, steps)
    Fores_moments.append(force_moment)
    print(Fores_moments)
    plot_force(np.array(Fores_moments).transpose(), steps_lis)
    # Env_model.plot_force(np.array(Fores_moments).transpose(), steps_lis)
    # Env_model.plot_figure(np.array(Fores_moments).transpose(), steps_lis, True)

plt.show(block=True)
# plot_force(np.array(Fores_moments).transpose(), steps)





