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

# Gymnasium environment to train RL agent
class MavEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # self.writer = None

        # Observation space: [lin_vel, ang_vel, gravity vector in body frame, fan_speeds, nozzle_angles]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * 18),
            high=np.array([np.inf] * 18),
            dtype=np.float32
        )
        
        # Actions space: 
        # [fanspeed1_setpoint_dot, fanspeed2_setpoint_dot, fanspeed3_setpoint_dot, #[PWM/s]
        # nozzleangle1_setpoint_dot, nozzleangle2_setpoint_dot, nozzleangle3_setpoint_dot, nozzleangle4_setpoint_dot, nozzleangle5_setpoint_dot, nozzleangle6_setpoint_dot #[rad/s]]
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
                              0, 0, 0,           # [19:22] fan speeds \in [0, 1]
                              0, 0, 0, 0, 0, 0,  # [22:28] nozzle angles [rad]
                              0, 0, 0,           # [28:31] fan speeds setpoints \in [0, 1]
                              0, 0, 0, 0, 0, 0]) # [31:37] nozzle angles setpoints 
        
        # Physical and simulation parameters
        self.mass = 5.218  # [kg]
        self.inertia = np.array([0.059829689, 0.06088309, 0.098981953])  # [kg*m^2], TODO: non-diagonal elements
        self.g = np.array([0, 0, -9.81])    # [m/s^2], Gravity vector in world frame
        self.k_f = 0.00005749               # [N/(PWM-1050)²], Thrust constant, Thrust_force = k_f * omega²
        self.k_phi = 6 / 4                  # [Hz], First order nozzle angle model, 1/tau where tau is time constant. TODO: 6 Hz is servo following setpoint. Gear ratio of 4 => nozzle_angle k is 6/4
        self.k_omega = 12 /2                # [Hz], First order fan speed model TODO this is actually k_forceomega TODO: 12 Hz is Force following setpoint. 12/2 is for pwm, but TODO!
        self.step_counter = 0
        self.dt = 0.01  # [s]
        
    # def set_writer(self, writer):
    #     self.writer = writer
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.step_counter = 0
        # Initialize state: 
        
        # Randomize position (x, y, z)
        position = np.array([0, 0, 0])

        # Randomize orientation quaternion [x, y, z, w] (ensure it's a valid quaternion)
        # rpy = np.random.uniform(low=-5, high=5, size=3)
        # orientation = R.as_quat(R.from_euler('xyz', rpy, degrees=True))
        orientation = np.array([0, 0, 0, 1])

        # Randomize linear and angular velocity and acceleration
        lin_vel = np.random.uniform(low=-0.0, high=0.0, size=3)
        ang_vel = np.random.uniform(low=-0.0, high=0.0, size=3)
        lin_acc = np.random.uniform(low=-0.0, high=0.0, size=3)
        ang_acc = np.random.uniform(low=-0.0, high=0.0, size=3)

        # Randomize actuators
        # fan_speeds = np.random.uniform(low=0.5, high=0.7, size=3)
        fan_speeds = np.array([0.61, 0.61, 0.61])
        nozzle_angles = np.array([0.80, -1.25, 0.80, -1.25, 0.80, -1.25]) 
        fanspeeds_setpoints = np.array([0.61, 0.61, 0.61])
        nozzle_setpoints = np.array([0.80, -1.25, 0.80, -1.25, 0.80, -1.25])


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
        
        g_bodyframe = quaternion_rotate_vector(orientation, self.g) # TODO: Is this correct? Shouldn't it be with inverse quaternion?
        obs = np.concatenate([lin_vel, ang_vel, g_bodyframe, fan_speeds, nozzle_angles])
        info = {"state": self.state}
        return obs, info
    
    # First order actuator models, k[Hz] = 1 / tau [s]
    # tau [s] is the empirical time constant of the actuator state following a setpoint
    # state_dot_k = (1/tau) * (setpoint_k - state_k)
    # state_{k+1} = state_k + state_dot_k * dT
    # action is change in actuator setpoint
    def first_order_actuator_models(self, action):
        nozzles_state = self.state[22:28]
        nozzles_setpoint = self.state[31:37]
        # Update setpoints and nozzle_angles [rad] according to first order model
        nozzles_setpoint += action[3:] / self.k_phi                             # Update setpoint with inverse 1° model
        nozzles_setpoint = nozzles_setpoint.clip(-3*np.pi, 3*np.pi)
        nozzles_dot = self.k_phi * (nozzles_setpoint - nozzles_state)           # TODO: nozzle_dot = self.k_phi_randomized * (nozzles_setpoint - nozzle_state)
        nozzles_state += nozzles_dot * self.dt

        fanspeeds_state = self.state[19:22]
        fanspeeds_setpoint = self.state[28:31]
        # Update setpoints and normalized fanspeeds according to first order model
        fanspeeds_setpoint += action[:3] / self.k_omega                          # Same as above
        fanspeeds_setpoint = fanspeeds_setpoint.clip(0, 1)
        fanspeeds_dot = self.k_omega * (fanspeeds_setpoint - fanspeeds_state)   # TODO: Same as above
        fanspeeds_state += fanspeeds_dot * self.dt
        fanspeeds_state = fanspeeds_state.clip(0, 1)

        return nozzles_state, nozzles_setpoint, fanspeeds_state, fanspeeds_setpoint

    # Compute thrust vectors [N] of the 3 nozzles in body frame
    def compute_thrust_vectors(self, nozzles_angles, fanspeeds):
        # thrust = k_f * (PWM - 1050)² * normal_vector
        fanspeeds_pwm = 1050 + fanspeeds * 900                                  # Scale normalized fanspeeds \in [0, 1] to PWM signal \in [1050, 1950]
        fanspeeds_squared = np.square(fanspeeds_pwm - 1050)[:, np.newaxis]
        thrust_vectors = self.k_f * fanspeeds_squared * thrustdirections(nozzles_angles)
        return thrust_vectors
    
    def compute_forces_and_torques(self, thrust_vectors):
        # Net force is sum of all thrust vectors
        force = np.sum(thrust_vectors, axis=0)
        
        # Compute torques from each thrust
        nozzle_states = np.zeros(6) # TODO use actual phi commands
        r_BE_1, r_BE_2, r_BE_3 = r_BE(nozzle_states)
        torque = np.zeros(3)
        torque += np.cross(r_BE_1, thrust_vectors[0])
        torque += np.cross(r_BE_2, thrust_vectors[1])
        torque += np.cross(r_BE_3, thrust_vectors[2])
        
        return force, torque
    
    def step(self, action):
        terminated = False
        truncated = False

        # Extract current state
        position = self.state[0:3]
        orientation = R.from_quat(self.state[3:7])
        lin_vel = self.state[7:10]
        ang_vel = self.state[10:13]

        # Update actuators
        nozzles_angles, nozzle_setpoints, fanspeeds, fanspeeds_setpoints = self.first_order_actuator_models(action)

        # Compute thrust vectors from actuator states [bodyframe]
        thrust_vectors = self.compute_thrust_vectors(nozzles_angles, fanspeeds)
        
        # Compute net force and torque [bodyframe]
        force, torque = self.compute_forces_and_torques(thrust_vectors)

        # Update angular velocity and orientation
        ang_acc = torque / self.inertia # TODO np.cross(ang_vel, self.inertia @ angular_velocity)
        ang_vel += ang_acc * self.dt
        orientation *= R.from_rotvec(ang_vel * self.dt)

        # Calculate gravity vector in body frame
        # orientation is body frame in world frame.
        # To calculate a world vector in body frame, orientation.inv() is used!
        g_bodyframe = orientation.inv().apply(self.g)
        
        # Update linear velocity and position
        lin_acc = force / self.mass + g_bodyframe # TODO NE eqt. np.cross(ang_vel, lin_vel)
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
            nozzles_angles,
            fanspeeds_setpoints,
            nozzle_setpoints
        ])

        obs = np.concatenate([lin_vel, ang_vel, g_bodyframe, fanspeeds, nozzles_angles])    # TODO: add fanspeeds_setpoint and nozzles_setpoint

        # Reward Function
        lin_vel_penalty = np.linalg.norm(lin_vel)
        ang_vel_penalty = np.linalg.norm(ang_vel)
        action_penalty = np.linalg.norm(action)
        fanspeed_penalty = np.linalg.norm(fanspeeds_setpoints - 0.61)
        nozzles_penalty = np.linalg.norm(nozzle_setpoints - np.array([0.80, -1.25, 0.80, -1.25, 0.80, -1.25]))
        reward_info = {"lin_vel_penalty": lin_vel_penalty, "ang_vel_penalty": ang_vel_penalty, "action_penalty": action_penalty}
        reward = 1 - 0.01 * lin_vel_penalty - 0.01 * ang_vel_penalty - 0.0 * action_penalty - 0.0 * fanspeed_penalty - 0.0 * nozzles_penalty

        # Terminate if velocity is going crazy or it rotates > 180°
        if (np.any(np.abs(lin_vel) > 5) or np.any(np.abs(ang_vel) > 5) or np.any(np.abs(orientation.as_euler('xyz', degrees=True) > 135))):
            reward -= 5
            terminated = True        
        
        # Truncate episode after 1000 timesteps
        self.step_counter += 1
        if (self.step_counter > 1_000):
            reward += 5
            truncated = True

        info = {"state": self.state, "reward": reward_info}
        
        return obs, reward, terminated, truncated, info
    


def train_MAV():

    env = MavEnv()

    # Uncomment to load model, not recommended
    # model = PPO.load("data/ppo_mav_model", env=env)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/ppo_mav/")

    eval_callback = TensorboardCallback(env=env, eval_freq=50_000, evaluate_fct=evaluate_model, verbose=1)

    model.learn(total_timesteps=500_000, callback=eval_callback)

    model.save("data/ppo_mav_model")

class TensorboardCallback(BaseCallback):
    def __init__(self, env, eval_freq, evaluate_fct, verbose=0):
        super().__init__(verbose)
        self.eval_env = env
        self.eval_freq = eval_freq
        self.evaluate_fct = evaluate_fct

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            if self.verbose > 0:
                print(f"\nEvaluating at step {self.n_calls}...")

            fig1, fig2 = self.evaluate_fct(self.model, self.eval_env)
            self.logger.record("plots/fig1", Figure(fig1, close=True), exclude=("stdout", "log", "json", "csv"))
            self.logger.record("plots/fig2", Figure(fig2, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close('all')
        return True    

def evaluate_model(model, env):
    # env = MavEnv()
    
    # model = PPO.load("data/ppo_mav_model")

    obs, info = env.reset()
    print(f"""Evaluating Model with inital state: 
          position: {info['state'][:3]} 
          orientation rpy: {(R.from_quat(info['state'][3:7]).as_euler('xyz', degrees=True))} 
          lin_vel: {info['state'][7:10]} 
          ang_vel: {info['state'][10:13]} 
          lin_acc: {info['state'][13:16]} 
          ang_acc: {info['state'][16:19]} 
          fanspeeds: {info['state'][19:22]} 
          nozzles_angles {info['state'][22:28]}""")

    # Record states and actions
    observations = [copy.deepcopy(obs)]
    infos = [copy.deepcopy(info)]
    actions = []
    rewards = []

    terminated, truncated = False, False

    # Run one episode
    while not (terminated or truncated):

        action, _states = model.predict(obs, deterministic=True)
        # TODO: Remove hardcoded action:

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
    train_MAV()
    
    # model = PPO.load("data/ppo_mav_model")
    # env = MavEnv()
    # evaluate_model(model, env)
    # plt.show()