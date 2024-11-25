import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

def plot_episode(observations, infos, actions, rewards):

    observations = np.array(observations)                   # [lin_vel, ang_vel, g_bodyframe]
    actions = np.array(actions)                             # [fanspeed_setpoints, nozzleangle_setpoints]
    states = np.array([info['state'] for info in infos])    # State vector
                                                            # [:3] position [m]      
                                                            # [3:7] orientation quaternion [x, y, z, w] of body frame in world frame
                                                            # [7:10] linear velocity [m/s]
                                                            # [10:13] angular velocity [rad/s]
                                                            # [13:16] linear acc [m/s²]
                                                            # [16:19] angular acc [rad/s²]  
                                                            # [19:22] fan speeds [PWM]
                                                            # [22:28] nozzle angles [rad]
    rewards = np.array(rewards)                             # rewards of episode

    fig, axs = plt.subplots(5, 2, figsize=(20, 12))

    axs[0,0].set_title("Position")
    axs[0,0].set_ylabel("[m]")
    axs[0,0].set_xlabel("timesteps")
    axs[0,0].plot(states[:,0], label=f"pos_x")
    axs[0,0].plot(states[:,1], label=f"pos_y")
    axs[0,0].plot(states[:,2], label=f"pos_z")
    axs[0,0].grid(True, alpha=0.3)
    axs[0,0].legend()

    axs[0,1].set_title("Orientation")
    axs[0,1].set_ylabel("[°]")
    axs[0,1].set_xlabel("timesteps")
    axs[0,1].set_yticks([-180, -90, 0, 90, 180])
    q = states[:, 3:7]
    rpy = R.from_quat(q).as_euler('xyz', degrees=True)
    axs[0,1].plot(rpy[:, 0], label=f"roll [°]")
    axs[0,1].plot(rpy[:, 1], label=f"pitch [°]")
    axs[0,1].plot(rpy[:, 2], label=f"yaw [°]")
    axs[0,1].grid(True, alpha=0.3)
    axs[0,1].legend()

    axs[1,0].set_title("Linear Velocity in bodyframe")
    axs[1,0].set_ylabel("[m/s]")
    axs[1,0].set_xlabel("timesteps")
    axs[1,0].plot(states[:, 7], label=f"lin_vel_x")
    axs[1,0].plot(states[:, 8], label=f"lin_vel_y")
    axs[1,0].plot(states[:, 9], label=f"lin_vel_z")
    axs[1,0].grid(True, alpha=0.3)
    axs[1,0].legend()


    axs[1,1].set_title("Angular Velocity in bodyframe")
    axs[1,1].set_ylabel("[rad/s]")
    axs[1,1].set_xlabel("timesteps")
    axs[1,1].plot(states[:, 10], label=f"ang_vel_x")
    axs[1,1].plot(states[:, 11], label=f"ang_vel_y")
    axs[1,1].plot(states[:, 12], label=f"ang_vel_z")
    axs[1,1].grid(True, alpha=0.3)
    axs[1,1].legend()

    axs[2,0].set_title("Linear Acceleration in bodyframe")
    axs[2,0].set_ylabel("[m/s²]")
    axs[2,0].set_xlabel("timesteps")
    axs[2,0].plot(states[:, 13], label=f"lin_acc_x")
    axs[2,0].plot(states[:, 14], label=f"lin_acc_y")
    axs[2,0].plot(states[:, 15], label=f"lin_acc_z")
    axs[2,0].grid(True, alpha=0.3)
    axs[2,0].legend()

    axs[2,1].set_title("Angular acceleration in bodyframe")
    axs[2,1].set_ylabel("[rad/s²]")
    axs[2,1].set_xlabel("timesteps")
    axs[2,1].plot(states[:, 16], label=f"ang_acc_x")
    axs[2,1].plot(states[:, 17], label=f"ang_acc_y")
    axs[2,1].plot(states[:, 18], label=f"ang_acc_z")
    axs[2,1].grid(True, alpha=0.3)
    axs[2,1].legend()

    basic_colors = plt.cm.tab10.colors 

    axs[3,0].set_title("Fanspeeds")
    axs[3,0].set_ylabel("[PWM]")
    axs[3,0].set_xlabel("timesteps")
    axs[3,0].set_ylim(1050, 1950)
    axs[3,0].set_yticks([1050, 1500, 1950])
    for i in range(19, 22, 1):
        axs[3,0].plot(1050 + 450 * (states[:, i] + 1), label=f"fanspeeds_{i-19}", color=basic_colors[i-19])
        axs[3,0].plot(1050 + 450 * (actions[:, i-19] + 1), label=f"fanspeeds_setpoints{i-19}", linestyle='--', color=basic_colors[i-19], alpha=0.7)
    axs[3,0].grid(True, alpha=0.3)
    axs[3,0].legend()

    axs[3,1].set_title("Nozzle angles")
    axs[3,1].set_ylabel("[°]")
    axs[3,1].set_xlabel("timesteps")
    axs[3,1].set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    for i in range(22, 28, 1):
        axs[3,1].plot(np.rad2deg(states[:, i]), label=f"nozzle_angles_{i-19}", color=basic_colors[i-22])
        axs[3,1].plot(np.rad2deg(actions[:, i-19]), label=f"nozzle_angles_setpoints{i-19}", linestyle='--', color=basic_colors[i-22], alpha=0.7)
    axs[3,1].grid(True, alpha=0.3)
    axs[3,1].legend()
    
    axs[4,0].set_title("Reward")
    axs[4,0].set_xlabel("timesteps")
    axs[4,0].plot(rewards, label="Reward", marker='.', linestyle='', markersize=3)
    axs[4,0].grid(True, alpha=0.3)
    axs[4,0].legend()

    axs[4,1].set_title("Gravity Vector in Bodyframe")
    axs[4,1].set_ylabel("[m/s²]")
    axs[4,1].set_xlabel("timesteps")
    axs[4,1].plot(observations[:, 6], label=f"g_bodyframe_x")
    axs[4,1].plot(observations[:, 7], label=f"g_bodyframe_y")
    axs[4,1].plot(observations[:, 8], label=f"g_bodyframe_z")
    axs[4,1].grid(True, alpha=0.3)
    axs[4,1].legend()

    plt.tight_layout()
    plt.show()