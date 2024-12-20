import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import time
from ForwardKinematics import thrustdirections, r_BE
from Quaternion import quaternion_rotate_vector
from Plots import plot_episode

# Gymnasium environment to train RL agent
class MavEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Observation space: [lin_vel, ang_vel, gravity vector in body frame]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * 9),
            high=np.array([np.inf] * 9),
            dtype=np.float32
        )
        
        # Actions space: 
        # [fanspeed1_setpoint, fanspeed2_setpoint, fanspeed3_setpoint, #[rad]
        # nozzleangle1_setpoint, nozzleangle2_setpoint, nozzleangle3_setpoint, nozzleangle4_setpoint, nozzleangle5_setpoint, nozzleangle6_setpoint #[PWM]]
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, -1, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi]),
            high=np.array([1, 1, 1, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi]),
            dtype=np.float32
        )

        # Initialize 21 states
        self.state = np.array([0, 0, 0,          # [:3] position [m]
                              0, 0, 0, 1,        # [3:7] orientation quaternion [x, y, z, w] of body frame in world frame
                              0, 0, 0,           # [7:10] linear velocity [m/s]
                              0, 0, 0,           # [10:13] angular velocity [rad/s]
                              0, 0, 0,           # [13:16] linear acc [m/s²]
                              0, 0, 0,           # [16:19] angular acc [rad/s²]
                              0, 0, 0,  # [19:22] fan speeds [PWM]
                              0, 0, 0, 0, 0, 0]) # [22:28] nozzle angles [rad]
        
        # Physical and simulation parameters
        self.mass = 5.218  # [kg]
        self.inertia = np.array([0.059829689, 0.06088309, 0.098981953])  # [kg*m^2], TODO: non-diagonal elements
        self.g = np.array([0, 0, -9.81])    # [m/s^2], Gravity vector in world frame
        self.k_f = 0.00005749               # [N/(PWM-1050)²], Thrust constant, Thrust_force = k_f * omega²
        self.k_phi = 6                      # [Hz], First order nozzle angle model, 1/tau where tau is time constant
        self.k_omega = 12                   # [Hz], First order fan speed model TODO this is actually k_forceomega
        self.step_counter = 0
        self.dt = 0.01  # [s]
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.step_counter = 0
        # Initialize state: 
        
        # Randomize position (x, y, z)
        position = np.random.uniform(low=-10, high=10, size=3)  # Example range [-10, 10] for each axis
        # position = np.array([0, 0, 0])

        # Randomize orientation quaternion [x, y, z, w] (ensure it's a valid quaternion)
        rpy = np.random.uniform(low=-30, high=30, size=3)
        orientation = R.as_quat(R.from_euler('xyz', rpy, degrees=True))
        # orientation = np.array([0, 0, 0, 1])

        # Randomize linear and angular velocity and acceleration
        lin_vel = np.random.uniform(low=-2, high=2, size=3)  # Example range [-5, 5]
        ang_vel = np.random.uniform(low=-2, high=2, size=3)  # Example range [-5, 5]
        lin_acc = np.random.uniform(low=-2, high=2, size=3)  # Example range [-2, 2]
        ang_acc = np.random.uniform(low=-2, high=2, size=3)  # Example range [-2, 2]

        # Randomize actuators
        fan_speeds = np.random.uniform(low=-0.5, high=0.5, size=3)  # Example range [0, 100]
        nozzle_angles = np.random.uniform(low=-1, high=1, size=6)  # Example range [-1.5, 1.5]

        # Combine all into state vector
        self.state = np.concatenate([
            position, 
            orientation, 
            lin_vel, 
            ang_vel, 
            lin_acc, 
            ang_acc, 
            fan_speeds, 
            nozzle_angles
        ])
        
        g_bodyframe = quaternion_rotate_vector(orientation, self.g)
        obs = np.concatenate([lin_vel, ang_vel, g_bodyframe])
        info = {"state": self.state}
        return obs, info
    
    # First order actuator models, k[Hz] = 1 / tau [s]
    # tau [s] is the empirical time constant of the actuator state following a setpoint [s]
    # state_dot_k = (1/tau) * (setpoint_k - state)
    # state_{k+1} = state_k + state_dot_k * dT
    def first_order_actuator_models(self, action):
        # Update nozzle angle [rad] according to first order model of error = setpoint - state
        nozzles_setpoint = action[3:]
        nozzles_state = self.state[22:28]
        nozzles_dot = self.k_phi * (nozzles_setpoint - nozzles_state)
        nozzles_state += nozzles_dot * self.dt

        # Update fan speed [PWM] according to first order model of error = setpoint - state
        fanspeeds_setpoint = action[:3]
        fanspeeds_state = self.state[19:22]
        fanspeeds_dot = self.k_omega * (fanspeeds_setpoint - fanspeeds_state)
        fanspeeds_state += fanspeeds_dot * self.dt

        return nozzles_state, fanspeeds_state

    # Compute thrust vectors [N] of the 3 nozzles in body frame
    def compute_thrust_vectors(self, nozzles_angles, fanspeeds):
        # thrust = k_f * (PWM - 1050)² * normal_vector
        # In this case, using the actual PWM signal, so -1050 is NOT already subtracted:
        fanspeeds = 1050 + (fanspeeds + 1) * 450
        fanspeeds_squared = np.square(fanspeeds-1050)[:, np.newaxis]
        thrust_vectors = self.k_f * fanspeeds_squared * thrustdirections(nozzles_angles)
        return thrust_vectors
    
    def compute_forces_and_torques(self, thrust_vectors):
        """Compute net force and torque from thrust vectors."""
        # Net force is sum of all thrust vectors
        force = np.sum(thrust_vectors, axis=0)
        
        # Compute torques from each thrust
        torque = np.zeros(3)
        action_phi = np.zeros(6) # TODO use actual phi commands
        r_BE_1, r_BE_2, r_BE_3 = r_BE(action_phi)
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
        nozzles_angles, fanspeeds = self.first_order_actuator_models(action)

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
            nozzles_angles
        ])

        obs = np.concatenate([lin_vel, ang_vel, g_bodyframe])

        # Terminate if velocity is going crazy
        if (np.any(np.abs(lin_vel) > 5) or np.any(np.abs(ang_vel) > 5)):
            terminated = True

        # Compute reward
        velocity_penalty = np.linalg.norm(lin_vel) + np.linalg.norm(ang_vel)
        
        # Gets 1 reward for every flying frame
        reward = 1 - 0.1 * velocity_penalty - 0.01 * np.linalg.norm(action)
        
        # Check if truncated
        self.step_counter += 1
        if (self.step_counter > 1_000):
            truncated = True

        info = {"state": self.state}
        
        return obs, reward, terminated, truncated, info
    


def train_MAV():

    env = MavEnv()

    model = PPO.load("data/ppo_mav_model", env=env)
    # Uncomment for new model
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/ppo_mav/")

    model.learn(total_timesteps=50_000)

    model.save("data/ppo_mav_model")


def evaluate_model():
    env = MavEnv()
    
    model = PPO.load("data/ppo_mav_model")

    obs, info = env.reset()
    print(f"""Evaluating Model with inital state: 
          position: {info['state'][:3]} 
          orientation: {info['state'][3:7]} 
          lin_vel: {info['state'][7:10]} 
          ang_vel: {info['state'][10:13]} 
          lin_acc: {info['state'][13:16]} 
          ang_acc: {info['state'][16:19]} 
          fanspeeds: {info['state'][19:22]} 
          nozzles_angles {info['state'][22:28]}""")

    # Record states and actions
    observations = [obs]
    infos = [info]
    actions = []
    rewards = []

    terminated, truncated = False, False
    
    # Run one episode
    while not (terminated or truncated):

        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        observations.append(obs)
        infos.append(info)
        actions.append(action)
        rewards.append(reward)
    
    plot_episode(observations, infos, actions, rewards)
    

if __name__ == "__main__":

    print(f"test_MAV")
    # train_MAV()
    evaluate_model()
    