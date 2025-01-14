import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import time
from ForwardKinematics import thrustdirections, r_BE
from Quaternion import quaternion_rotate_vector
from Plots import plot_episode
from torch.utils.tensorboard import SummaryWriter
import copy
import random
from collections import deque

# Gymnasium environment to train RL agent
class MavEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Observation space: [lin_vel, ang_vel, gravity vector in body frame, fan_speeds, nozzle_angles, thrust/mass, torque/inertia, last_action]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * 33),
            high=np.array([np.inf] * 33),
            dtype=np.float32
        )
        
        # Action space: 
        # [\dot{\omega}_1, \dot{\omega}_2, \dot{\omega}_3, #[Change in Throttle]]
        # \dot{\phi}_1, \dot{\phi}_2, \dot{\phi}_3, \dot{\phi}_4, \dot{\phi}_5, \dot{\phi}_6 # [1/s], normalized change in angle
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1]),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            dtype=np.float32
        )

        # Initialize 21 states
        self.state = np.array([0, 0, 0,          # [:3] position [m]
                              0, 0, 0, 1,        # [3:7] orientation quaternion [x, y, z, w] of body frame in world frame
                              0, 0, 0,           # [7:10] linear velocity [m/s]
                              0, 0, 0,           # [10:13] angular velocity [rad/s]
                              0, 0, 0,           # [13:16] linear acc [m/s²]
                              0, 0, 0,           # [16:19] angular acc [rad/s²]
                              0, 0, 0,           # [19:22] fanspeeds \in [0, 1]
                              0, 0, 0, 0, 0, 0,  # [22:28] nozzle angles [rad]
                              0, 0, 0,           # [28:31] fanspeeds setpoints \in [0, 1]
                              0, 0, 0, 0, 0, 0]) # [31:37] nozzle angles setpoints  [rad]
        
        # Physical and simulation parameters
        self.mass = 5.218  # [kg]
        self.inertia = np.array([0.059829689, 0.06088309, 0.098981953])  # [kg*m^2], TODO: non-diagonal elements
        self.g = np.array([0, 0, -9.81])    # [m/s^2], Gravity vector in world frame
        self.k_f = 0.00005749               # [N/(PWM-1050)²], Thrust constant, Thrust_force = k_f * omega²
        self.k_phi = 10.586                 # [Hz], First order nozzle angle model, 1/tau where tau is time constant
        self.k_phi_std = 3.04               # [Hz], standard deviation
        self.k_omega = 12                   # [Hz], First order fan speed model 
        self.k_omega_std = 1                # [Hz], Totally guessed
        self.phi_dot_max = 2
        self.omega_dot_max = 2
        self.vel_ref = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.step_counter = 0
        self.episode_length = 750
        self.dt = 0.01  # [s]
        self.action_buffer = deque([])
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.step_counter = 0
        self.episode_length = np.random.uniform(low=6000, high=6000)
        self.k_f = np.random.normal(0.00005749, 0.000005749)                # std dev is +-10%
        self.k_phi = np.random.normal(10.586, self.k_phi_std)
        self.k_omega = np.random.normal(12, self.k_omega_std)
        # self.vel_ref = np.random.uniform(low=-1, high=1, size=6)
        self.vel_ref = np.zeros(6)
        # Randomize time delays
        r = random.random()
        if r < 0.25:
            self.action_buffer = deque([np.zeros(9)])
        elif r < 0.5:
            self.action_buffer = deque([np.zeros(9), np.zeros(9)])
        elif r < 0.75:
            self.action_buffer = deque([np.zeros(9), np.zeros(9), np.zeros(9)])
        else:
            self.action_buffer = deque([np.zeros(9), np.zeros(9), np.zeros(9), np.zeros(9)])
        print(f"r = {r:.3f}, action_buffer_len:{len(self.action_buffer)}")
        # Initialize state: 
        
        # Randomize position (x, y, z)
        position = np.array([0, 0, 0])

        # Randomize orientation quaternion [x, y, z, w] (ensure it's a valid quaternion)
        rpy = np.random.uniform(low=-5, high=5, size=3)
        orientation = R.as_quat(R.from_euler('xyz', rpy, degrees=True))
        # orientation = np.array([0, 0, 0, 1])

        # Randomize linear and angular velocity and acceleration
        lin_vel = np.random.uniform(low=-0.1, high=0.1, size=3)
        ang_vel = np.random.uniform(low=-0.1, high=0.1, size=3)
        lin_acc = np.random.uniform(low=-0.1, high=0.1, size=3)         # Side note, this should be lin_acc = force / self.mass + g_bodyframe
        ang_acc = np.random.uniform(low=-0.1, high=0.1, size=3)         # and this ang_acc = torque / self.inertia

        # Randomize actuators
        fan_speeds = np.random.uniform(low=0.3, high=0.31, size=3)
        fanspeeds_setpoints = fan_speeds
        # nozzle_angles = np.random.uniform(low=-1, high=1, size=6)
        # RL Policy prefers this...
        nozzle_angles = np.array([0.80, -1.25, 0.81, -1.25, -0.79, +1.25]) + np.random.uniform(-0.1, 0.1, 6)
        nozzle_setpoints = nozzle_angles

        # Combine all into state vector
        self.state = np.concatenate([
            position, 
            orientation, 
            lin_vel, 
            ang_vel, 
            lin_acc, 
            ang_acc, 
            fan_speeds, 
            nozzle_angles,
            fanspeeds_setpoints,
            nozzle_setpoints
        ])


        thrust_vectors = self.compute_thrust_vectors(nozzle_angles, fan_speeds)
        force, torque = self.compute_forces_and_torques(thrust_vectors, nozzle_angles)
        
        # g_bodyframe = quaternion_rotate_vector(orientation, self.g)     # TODO: BUG! THIS IS WRONG! IT's ORIENTATIN INVERSE!
        g_bodyframe = R.from_euler('xyz', rpy, degrees=True).inv().apply(self.g)
        lin_vel_err = lin_vel - self.vel_ref[:3]
        ang_vel_err = ang_vel - self.vel_ref[3:]
        last_action = np.zeros(9)
        obs = np.concatenate([lin_vel_err, ang_vel_err, g_bodyframe, fan_speeds, nozzle_angles, force/self.mass - np.cross(ang_vel, lin_vel), torque/self.inertia, last_action])
        info = {"state": self.state, "k_f": self.k_f, "k_omega": self.k_omega, "k_phi": self.k_phi}
        return obs, info
    
    # First order actuator models, k[Hz] = 1 / tau [s]
    # tau [s] is the empirical time constant of the actuator state following a setpoint [s]
    # state_dot_k = (1/tau) * (setpoint_k - state)
    # state_{k+1} = state_k + state_dot_k * dT
    def first_order_actuator_models(self, action):
        # Update nozzle angle [rad] according to first order model of error = setpoint - state
        phi_dot_cmd = action[3:] * self.phi_dot_max 
        phi_state = self.state[22:28]
        phi_setpoint = phi_state + phi_dot_cmd / 10.586             # Using nominal k_phi value 10.586 Hz
        phi_dot = self.k_phi * (phi_setpoint - phi_state)
        phi_state += phi_dot * self.dt

        # Update fan speed [PWM] according to first order model of error = setpoint - state
        omega_dot_cmd = action[:3] * self.omega_dot_max
        omega_state = self.state[19:22]
        omega_setpoint = omega_state + omega_dot_cmd / 12.0         # Using nominal k_omega value of 12.0Hz
        omega_setpoint = np.clip(omega_setpoint, 0, 1)
        # if self.step_counter < 300:
        #     omega_setpoint = np.clip(omega_setpoint, 0.4, 1)
        omega_dot = self.k_omega * (omega_setpoint - omega_state)
        omega_state += omega_dot * self.dt
        omega_state = np.clip(omega_state, 0, 1)

        return phi_state, phi_setpoint, omega_state, omega_setpoint

    # Compute thrust vectors [N] of the 3 nozzles in body frame
    def compute_thrust_vectors(self, nozzles_angles, fanspeeds):
        # thrust = k_f * (PWM - 1050)² * normal_vector
        # In this case, using the actual PWM signal, so -1050 is NOT already subtracted:
        fanspeeds = 1050 + fanspeeds * 900
        fanspeeds_squared = np.square(fanspeeds-1050)[:, np.newaxis]
        thrust_vectors = self.k_f * fanspeeds_squared * thrustdirections(nozzles_angles)
        return thrust_vectors
    
    def compute_forces_and_torques(self, thrust_vectors, nozzle_angles):
        """Compute net force and torque from thrust vectors."""
        # Net force is sum of all thrust vectors
        force = np.sum(thrust_vectors, axis=0)

        # Compute distance vector r from Body frame to Endeffector of the Nozzle, where thrust is applied
        r_BE_1, r_BE_2, r_BE_3 = r_BE(nozzle_angles)

        # Compute torques from each thrust
        torque = np.zeros(3)
        torque += np.cross(r_BE_1, thrust_vectors[0])
        torque += np.cross(r_BE_2, thrust_vectors[1])
        torque += np.cross(r_BE_3, thrust_vectors[2])
        
        return force, torque
    
    def step(self, action):
        terminated = False
        truncated = False

        # # Time_dependent randomizations
        # self.k_f -= 0.000000057
        # self.vel_ref[5] = np.sin(0.01 * self.step_counter)
        # self.vel_ref[2] = np.sin(0.01 * self.step_counter)
        # Buffer actions to simulate time delays
        self.action_buffer.append(action)
        action = self.action_buffer.popleft()

        # Extract current state
        position = self.state[0:3]
        orientation = R.from_quat(self.state[3:7])
        lin_vel = self.state[7:10]
        ang_vel = self.state[10:13]

        # Update actuators
        nozzle_angles, nozzle_setpoints, fanspeeds, fanspeeds_setpoints = self.first_order_actuator_models(action)

        # Compute thrust vectors from actuator states [bodyframe]
        thrust_vectors = self.compute_thrust_vectors(nozzle_angles, fanspeeds)
        
        # Compute net force and torque [bodyframe]
        force, torque = self.compute_forces_and_torques(thrust_vectors, nozzle_angles)

        # Update angular velocity and orientation
        ang_acc = torque / self.inertia # TODO np.cross(ang_vel, self.inertia @ angular_velocity)
        ang_vel += ang_acc * self.dt
        orientation *= R.from_rotvec(ang_vel * self.dt)

        # Calculate gravity vector in body frame
        # orientation is body frame in world frame.
        # To calculate a world vector in body frame, orientation.inv() is used!
        g_bodyframe = orientation.inv().apply(self.g)
        
        # Update linear velocity and position
        lin_acc = force / self.mass + g_bodyframe - np.cross(ang_vel, lin_vel) # With Coriolis force
        lin_vel += lin_acc * self.dt
        position += orientation.apply(lin_vel) * self.dt
        
        # Update state
        self.state = np.concatenate([
            position,
            orientation.as_quat(),
            lin_vel,
            ang_vel,
            lin_acc,
            ang_acc,
            fanspeeds,
            nozzle_angles,
            fanspeeds_setpoints,
            nozzle_setpoints
        ])


        # Reward Function
        lin_vel_err = lin_vel - self.vel_ref[:3]
        lin_vel_penalty = np.linalg.norm(lin_vel_err)
        ang_vel_err = ang_vel - self.vel_ref[3:]
        ang_vel_penalty = np.linalg.norm(ang_vel_err)
        action_penalty = np.linalg.norm(action)
        # orientation_penalty = np.linalg.norm(orientation.as_euler('xyz', degrees=True))
        # fanspeed_penalty = np.linalg.norm(fanspeeds_setpoints - 0.61)
        # nozzles_penalty = np.linalg.norm(nozzle_setpoints - np.array([0.80, -1.25, 0.80, -1.25, 0.80, -1.25]))
        # setpoint_diff_penalty = np.linalg.norm(action - self.last_action)
        # turn_penalty = 0
        reward = - 0.12 * lin_vel_penalty - 0.01 * ang_vel_penalty - 0.005 * action_penalty
        reward_info = {"lin_vel_penalty": - 0.12 * lin_vel_penalty, "ang_vel_penalty": - 0.01 * ang_vel_penalty, "setpoint_diff_penalty": - 0.01 * action_penalty}
        
        obs = np.concatenate([lin_vel_err, ang_vel_err, g_bodyframe, fanspeeds, nozzle_angles, force/self.mass, torque/self.inertia, action])

        # Check if truncated
        self.step_counter += 1
        if (self.step_counter > self.episode_length):
            truncated = True

        # # Terminate if velocity is going crazy
        if (np.any(np.abs(lin_vel) > 100) or np.any(np.abs(ang_vel) > 100) or np.any(np.abs(lin_acc) > 1000)):
            reward = - (self.episode_length - self.step_counter)
            terminated = True

        info = {"state": self.state, "reward": reward_info, "vel_ref": self.vel_ref}
        
        return obs, reward, terminated, truncated, info
    


def train_MAV():

    env = MavEnv()

    # Uncomment to load model, not recommended
    # model = PPO.load("data/ppo_28", env=env, tensorboard_log="./logs/09accPenalty/")
    model = PPO("MlpPolicy", env, learning_rate=lr_schedule, verbose=1, tensorboard_log="./logs/10actionBuffer/")

    eval_callback = TensorboardCallback(env=env, eval_freq=100_000, evaluate_fct=evaluate_model, verbose=1)

    model.learn(total_timesteps=3_500_000, callback=eval_callback)

    model.save("data/ppo_mav_model")

def lr_schedule(progress_remaining: float) -> float:
    return 0.0002 + progress_remaining * 0.0001  # Example: linear decay

class TensorboardCallback(BaseCallback):
    def __init__(self, env, eval_freq, evaluate_fct, verbose=0):
        super().__init__(verbose)
        self.eval_env = env
        self.eval_freq = eval_freq
        self.evaluate_fct = evaluate_fct
        self.save_path = "./logs/ppo_mav/"
        self.best_reward = -float("inf")  # Initialize with a very low value

    def _on_training_start(self):
        writer = SummaryWriter(log_dir=self.logger.dir)
        with open("src/GymEnvironment.py", "r") as f:
            code_content = f.read()

        writer.add_text("GymEnvironment.py", f"```\n{code_content}```", global_step=0)
        writer.close()

    def _on_step(self) -> bool:
        # Evaluate model and plot on tensorboard
        if self.n_calls % self.eval_freq == 0:
            if self.verbose > 0:
                print(f"\nEvaluating at step {self.n_calls}...")

            fig1, fig2 = self.evaluate_fct(self.model, self.eval_env)
            self.logger.record("plots/fig1", Figure(fig1, close=True), exclude=("stdout", "log", "json", "csv"))
            self.logger.record("plots/fig2", Figure(fig2, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close('all')
        return True
    
    # def _on_rollout_end(self) -> None:
    #     # Evaluate the current rollout reward
    #     rollout_rewards = self.locals["rollout_buffer"].rewards
    #     mean_reward = np.mean(rollout_rewards)
    #     print(f"mean_reward: {mean_reward}, len: {len(rollout_rewards)}")
    #     if mean_reward > self.best_reward:
    #         self.best_reward = mean_reward
    #         if self.verbose > 0:
    #             print(f"New best reward: {self.best_reward:.2f}, saving model to {self.save_path}")
    #         # Save the best model
    #         self.model.save(f"{self.save_path}ppo_mav_{mean_reward:.3f}")

def evaluate_model(model, env):

    obs, info = env.reset()
    print(f"""Evaluating Model with inital state: 
          position: {info['state'][:3]} 
          orientation rpy: {(R.from_quat(info['state'][3:7]).as_euler('xyz', degrees=True))} 
          lin_vel: {info['state'][7:10]} 
          ang_vel: {info['state'][10:13]} 
          lin_acc: {info['state'][13:16]} 
          ang_acc: {info['state'][16:19]} 
          fanspeeds: {info['state'][19:22]} 
          nozzleangles {info['state'][22:28]}""")

    # Record states and actions
    observations = [copy.deepcopy(obs)]
    infos = [copy.deepcopy(info)]
    actions = []
    rewards = []

    terminated, truncated = False, False

    # Run one episode
    while not (terminated or truncated):

        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        # TODO: Nastyyy bug where appending dictionairy info to infos messes up everything.
        # Solved with copy.deepcopy()
        observations.append(copy.deepcopy(obs))
        infos.append(copy.deepcopy(info))
        actions.append(copy.deepcopy(action))
        rewards.append(copy.deepcopy(reward))

    env.close()
    fig1, fig2 = plot_episode(observations, infos, actions, rewards)
    return fig1, fig2

if __name__ == "__main__":

    print(f"test_MAV")
    # train_MAV()
    
    model = PPO.load("data/ppo_38")
    env = MavEnv()
    evaluate_model(model, env)
    plt.show()
    