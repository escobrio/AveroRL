import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

def plot_episode(observations, infos, actions, rewards):
    observations = np.array(observations)                   # [lin_vel, ang_vel, g_bodyframe]
    actions = np.array(actions)                             # [fanspeed_setpoints, nozzleangle_setpoints]
    rewards = np.array(rewards)                             # rewards of episode
    states = np.array([info['state'] for info in infos])    # State vector
                                                            # [:3] position [m]      
                                                            # [3:7] orientation quaternion [x, y, z, w] of body frame in world frame
                                                            # [7:10] linear velocity [m/s]
                                                            # [10:13] angular velocity [rad/s]
                                                            # [13:16] linear acc [m/s²]
                                                            # [16:19] angular acc [rad/s²]  
                                                            # [19:22] fan speeds [PWM]
                                                            # [22:28] nozzle angles [rad]
    lin_vel_penalty = np.array([info.get('reward', {}).get('lin_vel_penalty', 0) for info in infos])
    ang_vel_penalty = np.array([info.get('reward', {}).get('ang_vel_penalty', 0) for info in infos])
    setpoint_diff_penalty = np.array([info.get('reward', {}).get('setpoint_diff_penalty', 0) for info in infos])
    vel_ref = np.array([info.get('vel_ref', np.zeros(6)) for info in infos])

    k_f = infos[0]["k_f"]
    k_omega = infos[0]["k_omega"]
    k_phi = infos[0]["k_phi"]

    palette = plt.cm.Set1.colors
    # red, green, blue for x, y, z
    colors = [palette[0], palette[2], palette[1]] + list(palette[3:5]) + list(palette[7:])
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 2, hspace=0)
    axs = gs.subplots(sharex=True)

    axs[0,0].set_title("Position in worldframe", loc='center', y=0.85)
    axs[0,0].set_ylabel("[m]")
    axs[0,0].set_xlabel("timesteps")
    axs[0,0].plot(states[:,0], color=colors[0], label=f"pos_x", alpha=0.3)
    axs[0,0].plot(states[:,1], color=colors[1], label=f"pos_y", alpha=0.3)
    axs[0,0].plot(states[:,2], color=colors[2], label=f"pos_z", alpha=0.3)
    axs[0,0].plot(states[:,0], '.', markersize=1, color=colors[0])
    axs[0,0].plot(states[:,1], '.', markersize=1, color=colors[1])
    axs[0,0].plot(states[:,2], '.', markersize=1, color=colors[2])
    axs[0,0].grid(True, alpha=0.3)
    axs[0,0].axhline(0, color='black', alpha=0.2)
    axs[0,0].legend(loc='center right')


    axs[0,1].set_title("Orientation of body in worldframe", loc='center', y=0.85)
    axs[0,1].set_ylabel("[°]")
    axs[0,1].set_xlabel("timesteps")
    # axs[0,1].set_yticks(np.arange(-360, 360, 5))
    q = states[:, 3:7]
    rpy = R.from_quat(q).as_euler('xyz', degrees=True)
    axs[0,1].plot(rpy[:, 0], color=colors[0], label=f"roll [°]", alpha=0.3)
    axs[0,1].plot(rpy[:, 1], color=colors[1], label=f"pitch [°]", alpha=0.3)
    axs[0,1].plot(rpy[:, 2], color=colors[2], label=f"yaw [°]", alpha=0.3)
    axs[0,1].plot(rpy[:, 0], '.', markersize=1, color=colors[0])
    axs[0,1].plot(rpy[:, 1], '.', markersize=1, color=colors[1])
    axs[0,1].plot(rpy[:, 2], '.', markersize=1, color=colors[2])
    axs[0,1].grid(True, alpha=0.3)
    axs[0,1].axhline(0, color='black', alpha=0.2)
    axs[0,1].legend(loc='center right')

    axs[1,0].set_title("Linear Velocity in bodyframe", loc='center', y=0.85)
    axs[1,0].set_ylabel("[m/s]")
    axs[1,0].set_xlabel("timesteps")
    axs[1,0].axhline(0, color='black', alpha=0.2)
    axs[1,0].plot(states[:, 7], color=colors[0], label=f"lin_vel_x", alpha=0.3)
    axs[1,0].plot(states[:, 8], color=colors[1], label=f"lin_vel_y", alpha=0.3)
    axs[1,0].plot(states[:, 9], color=colors[2], label=f"lin_vel_z", alpha=0.3)
    axs[1,0].plot(states[:, 7], '.', markersize=1, color=colors[0])
    axs[1,0].plot(states[:, 8], '.', markersize=1, color=colors[1])
    axs[1,0].plot(states[:, 9], '.', markersize=1, color=colors[2])
    axs[1,0].plot(vel_ref[:, 0], color=colors[0], label=f"lin_vel_x_ref", alpha=0.3)
    axs[1,0].plot(vel_ref[:, 1], color=colors[1], label=f"lin_vel_y_ref", alpha=0.3)
    axs[1,0].plot(vel_ref[:, 2], color=colors[2], label=f"lin_vel_z_ref", alpha=0.3)
    axs[1,0].grid(True, alpha=0.3)
    axs[1,0].legend(loc='center right')


    axs[1,1].set_title("Angular Velocity in bodyframe", loc='center', y=0.85)
    axs[1,1].set_ylabel("[rad/s]")
    axs[1,1].set_xlabel("timesteps")
    axs[1,1].plot(states[:, 10], color=colors[0], label=f"ang_vel_x", alpha=0.3)
    axs[1,1].plot(states[:, 11], color=colors[1], label=f"ang_vel_y", alpha=0.3)
    axs[1,1].plot(states[:, 12], color=colors[2], label=f"ang_vel_z", alpha=0.3)
    axs[1,1].plot(states[:, 10], '.', markersize=1, color=colors[0])
    axs[1,1].plot(states[:, 11], '.', markersize=1, color=colors[1])
    axs[1,1].plot(states[:, 12], '.', markersize=1, color=colors[2])
    axs[1,1].plot(vel_ref[:, 3], color=colors[0], label=f"ang_vel_x_ref", alpha=0.3)
    axs[1,1].plot(vel_ref[:, 4], color=colors[1], label=f"ang_vel_y_ref", alpha=0.3)
    axs[1,1].plot(vel_ref[:, 5], color=colors[2], label=f"ang_vel_z_ref", alpha=0.3)
    axs[1,1].grid(True, alpha=0.3)
    axs[1,1].axhline(0, color='black', alpha=0.2)
    axs[1,1].legend(loc='center right')

    axs[2,0].set_title("Linear Acceleration in bodyframe", loc='center', y=0.85)
    axs[2,0].set_ylabel("[m/s²]")
    axs[2,0].set_xlabel("timesteps")
    axs[2,0].plot(states[:, 13], color=colors[0], label=f"lin_acc_x", alpha=0.3)
    axs[2,0].plot(states[:, 14], color=colors[1], label=f"lin_acc_y", alpha=0.3)
    axs[2,0].plot(states[:, 15], color=colors[2], label=f"lin_acc_z", alpha=0.3)
    axs[2,0].plot(states[:, 13], '.', markersize=1, color=colors[0])
    axs[2,0].plot(states[:, 14], '.', markersize=1, color=colors[1])
    axs[2,0].plot(states[:, 15], '.', markersize=1, color=colors[2])
    axs[2,0].grid(True, alpha=0.3)
    axs[2,0].axhline(0, color='black', alpha=0.2)
    axs[2,0].legend(loc='center right')

    axs[2,1].set_title("Angular acceleration in bodyframe", loc='center', y=0.85)
    axs[2,1].set_ylabel("[rad/s²]")
    axs[2,1].set_xlabel("timesteps")
    axs[2,1].plot(states[:, 16], color=colors[0], label=f"ang_acc_x", alpha=0.3)
    axs[2,1].plot(states[:, 17], color=colors[1], label=f"ang_acc_y", alpha=0.3)
    axs[2,1].plot(states[:, 18], color=colors[2], label=f"ang_acc_z", alpha=0.3)
    axs[2,1].plot(states[:, 16], '.', markersize=1, color=colors[0])
    axs[2,1].plot(states[:, 17], '.', markersize=1, color=colors[1])
    axs[2,1].plot(states[:, 18], '.', markersize=1, color=colors[2])
    axs[2,1].grid(True, alpha=0.3)
    axs[2,1].axhline(0, color='black', alpha=0.2)
    axs[2,1].legend(loc='center right')
    
    axs[3,0].set_title("Reward", loc='center', y=0.85)
    axs[3,0].set_xlabel("timesteps")
    axs[2,0].axhline(0, color='black', alpha=0.2)
    axs[3,0].plot(lin_vel_penalty, label="lin_vel_penalty", marker='.', linestyle='', markersize=3, alpha=0.3)
    axs[3,0].plot(ang_vel_penalty, label="ang_vel_penalty", marker='.', linestyle='', markersize=3, alpha=0.3)
    axs[3,0].plot(setpoint_diff_penalty, label="setpoint_diff_penalty", marker='.', linestyle='', markersize=3, alpha=0.3)
    axs[3,0].plot(rewards, label="Total Reward", marker='.', linestyle='', markersize=3)
    axs[3,0].grid(True, alpha=0.3)
    axs[3,0].legend(loc='center right')

    axs[3,1].set_title("Gravity Vector in Bodyframe", loc='center', y=0.85)
    axs[3,1].set_ylabel("[m/s²]")
    axs[3,1].set_xlabel("timesteps")
    axs[3,1].plot(observations[:, 6], color=colors[0], label=f"g_bodyframe_x", alpha=0.3)
    axs[3,1].plot(observations[:, 7], color=colors[1], label=f"g_bodyframe_y", alpha=0.3)
    axs[3,1].plot(observations[:, 8], color=colors[2], label=f"g_bodyframe_z", alpha=0.3)
    axs[3,1].plot(observations[:, 6], '.', markersize=1, color=colors[0])
    axs[3,1].plot(observations[:, 7], '.', markersize=1, color=colors[1])
    axs[3,1].plot(observations[:, 8], '.', markersize=1, color=colors[2])
    axs[3,1].grid(True, alpha=0.3)
    axs[3,1].axhline(0, color='black', alpha=0.2)
    axs[3,1].legend(loc='center right')
    plt.tight_layout()

    fig2 = plt.figure(figsize=(20, 12))
    gs2 = fig2.add_gridspec(3, 2, hspace=0)
    axs2 = gs2.subplots(sharex=True)

    axs2[0,0].set_title(fr"$Fanspeeds, k_\omega = {{{k_omega:.1f}}}Hz, k_f = {{{k_f:.2e}}} [N/PWM²]$", loc='center', y=0.91)
    axs2[0,0].set_ylabel("[0, 1]")
    axs2[0,0].set_xlabel("timesteps")
    axs2[0,0].set_ylim(0, 1)
    # axs2[0,0].set_ylim(1050, 1950)
    # axs2[0,0].set_yticks([1050, 1500, 1950])
    for i in range(19, 22, 1):
        # axs2[0,0].plot(1050 + 900 * states[:, i], 'o', markersize=1, color=colors[i-19], label=f"fanspeed_{i-19}")
        # axs2[0,0].plot(1050 + 900 * states[:, i+9], '.', markersize=2, linestyle='--', color=colors[i-19], label=f"fanspeed_setpoint{i-19}", alpha=0.5)
        axs2[0,0].plot(states[:, i], color=colors[i-19], label=fr"$\omega_{{{i-18}}}$", alpha=0.5)
        axs2[0,0].plot(states[:, i], '.', markersize=2, color=colors[i-19])
        axs2[0,0].plot(states[:, i+9], '.', markersize=2, linestyle='--', color=colors[i-19], label=fr"$\omega^*_{{{i-18}}}$", alpha=0.3)
    axs2[0,0].grid(True, alpha=0.3)
    axs2[0,0].legend(loc='center right')

    axs2[1,0].set_title("Fanspeeds Actions", loc='center', y=0.94)
    axs2[1,0].set_ylabel("[0, 1]")
    axs2[1,0].set_xlabel("timesteps")
    axs2[1,0].set_ylim(-1, 1)
    for i in range(0, 3, 1):
        axs2[1,0].plot(actions[:, i], color=colors[i], label=f"action_{i}", alpha=0.5)
        axs2[1,0].plot(actions[:, i], '.', markersize=2, color=colors[i])
    axs2[1,0].grid(True, alpha=0.3)
    axs2[1,0].axhline(0, color='black', alpha=0.2)
    axs2[1,0].legend(loc='center right')

    axs2[2,0].set_title(fr"$\dot\omega$", loc='center', y=0.91)
    axs2[2,0].set_ylabel("[0, 1]")
    axs2[2,0].set_xlabel("timesteps")
    axs2[2,0].set_ylim(-3, 3)
    # axs2[2,0].set_yticks([1050, 1500, 1950])
    for i in range(19, 22, 1):
        # axs2[2,0].plot(1050 + 900 * states[:, i], 'o', markersize=1, color=colors[i-19], label=f"fanspeed_{i-19}")
        # axs2[2,0].plot(1050 + 900 * states[:, i+9], '.', markersize=2, linestyle='--', color=colors[i-19], label=f"fanspeed_setpoint{i-19}", alpha=0.5)
        axs2[2,0].plot(np.diff(states[:, i]) / 0.01, color=colors[i-19], label=fr"$\dot\omega_{{{i-18}}}$", alpha=0.5)
        axs2[2,0].plot(np.diff(states[:, i]) / 0.01, '.', markersize=2, color=colors[i-19])
        axs2[2,0].plot(np.diff(states[:, i+9]) / 0.01, '.', markersize=2, linestyle='--', color=colors[i-19], label=fr"$\dot\omega^*_{{{i-18}}}$", alpha=0.3)
    axs2[2,0].grid(True, alpha=0.3)
    axs2[2,0].legend(loc='center right')

    axs2[0,1].set_title(fr"$Nozzle angles, k_\varphi = {{{k_phi:.1f}}} [Hz]$", loc='center', y=0.92)
    axs2[0,1].set_ylabel("[°]")
    axs2[0,1].set_xlabel("timesteps")
    # axs2[0,1].set_yticks(np.arange(-360, 360, 30))
    for i in range(3):
        for j in range(2):
            axs2[0,1].plot(np.rad2deg(states[:, 22+i*2+j]), color=colors[i*2+j], label=fr"$\varphi_{{{i+1}{j+1}}}$", alpha=0.5)
            axs2[0,1].plot(np.rad2deg(states[:, 22+i*2+j]), '.', markersize=2, color=colors[i*2+j])
            axs2[0,1].plot(np.rad2deg(states[:, 31+i*2+j]), '.', markersize=2, linestyle='--', color=colors[i*2+j], label=fr"$\varphi^*_{{{i+1}{j+1}}}$", alpha=0.2)
    axs2[0,1].grid(True, alpha=0.3)
    axs2[0,1].axhline(0, color='black', alpha=0.2)
    axs2[0,1].legend(loc='center right')

    axs2[1,1].set_title("Nozzle Actions", loc='center', y=0.93)
    axs2[1,1].set_ylabel("[0, 1]")
    axs2[1,1].set_xlabel("timesteps")
    axs2[1,1].set_ylim(-1, 1)
    for i in range(3, 9, 1):
        axs2[1,1].plot(actions[:, i], color=colors[i-3], label=f"action_{i}", alpha=0.5)
        axs2[1,1].plot(actions[:, i], '.', markersize=2, color=colors[i-3], alpha=1)
    axs2[1,1].grid(True, alpha=0.3)
    axs2[1,1].axhline(0, color='black', alpha=0.2)
    axs2[1,1].legend(loc='center right')

    axs2[2,1].set_title(fr"$\dot\varphi$", loc='center', y=0.92)
    axs2[2,1].set_ylabel("[rad/s]")
    axs2[2,1].set_xlabel("timesteps")
    # axs2[2.1].set_yticks(np.arange(-360, 360, 30))
    axs2[2,1].set_ylim(-3, 3)
    for i in range(3):
        for j in range(2):
            axs2[2,1].plot(np.diff(states[:, 22+i*2+j]) / 0.01, color=colors[i*2+j], label=fr"$\dot\varphi_{{{i+1}{j+1}}}$", alpha=0.5)
            axs2[2,1].plot(np.diff(states[:, 22+i*2+j]) / 0.01, '.', markersize=2, color=colors[i*2+j])
            axs2[2,1].plot(np.diff(states[:, 31+i*2+j]) / 0.01, '.', markersize=2, linestyle='--', color=colors[i*2+j], label=fr"$\dot\varphi^*_{{{i+1}{j+1}}}$", alpha=0.2)
    axs2[2,1].grid(True, alpha=0.3)
    axs2[2,1].axhline(0, color='black', alpha=0.2)
    axs2[2,1].legend(loc='center right')


    plt.tight_layout()
    # plt.show()
    return fig, fig2