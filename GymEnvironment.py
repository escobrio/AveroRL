import numpy as np
from scipy.spatial.transform import Rotation
import gymnasium as gym
from ForwardKinematics import thrustdirections, r_BE
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

class MavEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Every Gym Environment must have observation_space
        # State: [dx, dy, dz, droll, dpitch, dyaw, TODO gravity vector in body frame]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * 6),
            high=np.array([np.inf] * 6),
            dtype=np.float32
        )
        
        # Every Gym Environment must have action_space
        # Actions: [EDF1_des, EDF2_des, EDF3_des, servo1_des, servo2_des, servo3_des, servo4_des, servo5_des, servo6_des]
        self.action_space = gym.spaces.Box(
            low=np.array([1050, 1050, 1050, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi]),
            high=np.array([1950, 1950, 1950, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi]),
            dtype=np.float32
        )

        # Initialize 21 states
        self.state = np.array([0, 0, 1,  # [:3] position
                              0, 0, 0,  # [3:6] orientation
                              0, 0, 0,  # [6:9] linear velocity
                              0, 0, 0,  # [9:12]angular velocity
                              1050, 1050, 1050, # [12:15] fan speeds PWM
                              0, 0, 0, 0, 0, 0, # [15:21] nozzle angles
                              0, 0, 0,  # [21:24] linear acc
                              0, 0, 0]) # [24:27] angular acc
        
        # Physical parameters
        self.mass = 5.218  # kg
        self.inertia = np.array([0.059829689, 0.06088309, 0.098981953])  # kg*m^2
        self.dt = 0.016667  # s
        self.g = np.array([0, 0, 9.81])  # m/s^2
        self.k_f = 0.00005749 # Thrust constant, Thrust_force = k_f * omega²
        self.k_phi = 6 # Hz, First order nozzle angle model, 1/tau where tau is time constant
        self.k_omega = 12 # Hz, First order fan speed model TODO this is actually k_forceomega
        self.step_counter = 0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Initialize state: 
        # TODO: randomize
        self.state = np.array([0, 0, 0,  # position
                              0, 0, 0,  # orientation
                              0, 0, 0,  # linear velocity
                              0, 0, 0,  # angular velocity
                              545, 545, 545, # fan speeds
                              0.8, -1.25, 0.8, -1.25, 0.8, -1.25, # nozzle angles
                              0, 0, 0, # linear acceleration
                              0, 0, 0]) # angular acceleration
        obs = self.state[6:12]
        self.step_counter = 0
        info = self.state
        return obs, info
    
    def first_order_actuator_models(self, action):
        phi_des = action[3:]
        phi_state = self.state[15:21]
        phi_dot = self.k_phi * (phi_des - phi_state)
        phi_state += phi_dot * self.dt

        omega_des = action[:3]
        omega_state = self.state[12:15]
        omega_dot = self.k_omega * (omega_des - omega_state)
        omega_state += omega_dot * self.dt

        return phi_state, omega_state

    def compute_thrust_vectors(self, phi_state, omega_state):
        # thrust = k_f * (PWM - 1050)² * normal_vector
        # In this case, using the commanded PWM signal, so -1050 is already subtracted:
        omega_squared = np.square(omega_state)[:, np.newaxis]
        thrust_vectors = self.k_f * omega_squared * thrustdirections(phi_state)
        # print(f"\nk_f: {self.k_f} \nomega_squared: \n{omega_squared}, \nthrustdirections(phi_state): \n{thrustdirections(phi_state)} \n thrust_vectors: \n{thrust_vectors}")
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
        done = False
        # Update actuators
        phi_state, omega_state = self.first_order_actuator_models(action)

        # Get thrust vectors from EDF and servo settings
        thrust_vectors = self.compute_thrust_vectors(phi_state, omega_state)
        
        # Compute net force and torque
        force, torque = self.compute_forces_and_torques(thrust_vectors)
        
        # Extract current state
        position = self.state[0:3]
        orientation = self.state[3:6]
        linear_vel = self.state[6:9]
        angular_vel = self.state[9:12]
        
        # Create rotation matrix from current orientation
        R = Rotation.from_euler('xyz', orientation).as_matrix()
        
        # Update linear velocity and position
        # TODO calculate gravity vector in body frame
        linear_acc = force / self.mass - self.g # TODO NE eqt. np.cross(angular_vel, linear_vel)
        linear_vel += linear_acc * self.dt
        position += linear_vel * self.dt
        
        # Update angular velocity and orientation
        angular_acc = torque / self.inertia # TODO np.cross(angular_vel, self.inertia @ angular_velocity)
        angular_vel += angular_acc * self.dt
        orientation += angular_vel * self.dt # TODO calculate quaternion orientation from angular_vel
        
        # Update state
        self.state = np.concatenate([
            position,
            orientation,
            linear_vel,
            angular_vel,
            omega_state,
            phi_state,
            linear_acc,
            angular_acc
        ])

        obs = self.state[6:12]

        # Compute reward
        velocity_penalty = np.linalg.norm(linear_vel) + np.linalg.norm(angular_vel)
        
        reward = -velocity_penalty
        
        # Check if done
        self.step_counter += 1
        if (self.step_counter > 1000):
            done = True

        info = self.state
        
        return obs, reward, done, done, info
    
def plot_info(observations, infos, actions):

    observations = np.array(observations)
    infos = np.array(infos)
    actions = np.array(actions)

    fig, axs = plt.subplots(4, 2, figsize=(20, 12))

    for i in range(0, 3, 1):
        axs[0,0].plot(infos[:,i], label=f"position {i}")
        axs[0,0].legend()

    for i in range(3, 6, 1):
        axs[0,1].plot(infos[:,i], label=f"orientation {i}")
        axs[0,1].legend()

    for i in range(6, 9, 1):
        axs[1,0].plot(infos[:,i], label=f"lin_vel {i}")
        axs[1,0].legend()

    for i in range(9, 12, 1):
        axs[1,1].plot(infos[:,i], label=f"ang_vel {i}")
        axs[1,1].legend()

    for i in range(21, 24, 1):
        axs[2,0].plot(infos[:,i], label=f"lin_acc {i}")
        axs[2,0].legend()

    for i in range(24, 27, 1):
        axs[2,1].plot(infos[:,i], label=f"ang_acc {i}")
        axs[2,1].legend()

    for i in range(12, 15, 1):
        axs[3,0].plot(infos[:,i], label=f"fan_speeds {i}")
        axs[3,0].plot(actions[:,i-12], label=f"fan_speeds actions{i}")
        axs[3,0].legend()

    for i in range(15, 21, 1):
        axs[3,1].plot(infos[:,i], label=f"nozzle_angles {i}")
        axs[3,1].plot(actions[:,i-12], label=f"nozzle_angles actions{i}", linestyle='--')
        axs[3,1].legend()
    
    plt.tight_layout()
    plt.show()

def train_MAV():
    # vec_env = make_vec_env(lambda: MavEnv(), n_envs=1)
    env = MavEnv()

    model = PPO("MlpPolicy", env, verbose=1)

    eval_env = MavEnv()
    # eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/", log_path="./logs/", eval_freq=1000, deterministic=True, render=False)

    model.learn(total_timesteps=25_000)

    model.save("ppo_mav_model")


def evaluate_model():
    env = MavEnv()
    
    model = PPO.load("ppo_mav_model")

    obs, info = env.reset()
    # Record states and actions
    observations = [obs]
    infos = [info]
    actions = []
    
    for i in range(1000):

        action, _states = model.predict(obs, deterministic=True)

        obs, reward, done, done, info = env.step(action)

        observations.append(obs)
        infos.append(info)
        actions.append(action)

    # observations = np.array(observations)
    # observations = np.array([tup[0] for tup in observations])
    # print(f"observations: {observations}\nactions: {actions[11]}")
    
    plot_info(observations, infos, actions)
    

if __name__ == "__main__":

    print(f"test_MAV")
    # train_MAV()
    evaluate_model()
    