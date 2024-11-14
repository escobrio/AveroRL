import numpy as np
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
from ForwardKinematics import thrustdirections, r_BE
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

class MavEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # State: [dx, dy, dz, droll, dpitch, dyaw, TODO gravity vector in body frame]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * 6),
            high=np.array([np.inf] * 6),
            dtype=np.float32
        )
        
        # Actions: [EDF1_des, EDF2_des, EDF3_des, servo1_des, servo2_des, servo3_des, servo4_des, servo5_des, servo6_des]
        self.action_space = gym.spaces.Box(
            low=np.array([1050, 1050, 1050, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi]),
            high=np.array([1950, 1950, 1950, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi]),
            dtype=np.float32
        )

        # Initialize 21 states
        self.state = np.array([0, 0, 0,          # [:3] position [m]
                              0, 0, 0, 1,        # [3:7] orientation quaternion [x, y, z, w]
                              0, 0, 0,           # [7:10] linear velocity [m/s]
                              0, 0, 0,           # [10:13] angular velocity [rad/s]
                              0, 0, 0,           # [13:16] linear acc [m/s²]
                              0, 0, 0,           # [16:19] angular acc [rad/s²]
                              1050, 1050, 1050,  # [19:22] fan speeds [PWM]
                              0, 0, 0, 0, 0, 0]) # [22:28] nozzle angles [rad]
        
        # Physical parameters
        self.mass = 5.218  # kg
        self.inertia = np.array([0.059829689, 0.06088309, 0.098981953])  # kg*m^2
        self.dt = 0.01  # s
        self.g = np.array([0, 0, -9.81])  # m/s^2
        self.k_f = 0.00005749 # Thrust constant, Thrust_force = k_f * omega²
        self.k_phi = 6 # Hz, First order nozzle angle model, 1/tau where tau is time constant
        self.k_omega = 12 # Hz, First order fan speed model TODO this is actually k_forceomega
        self.step_counter = 0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Initialize state: 
        # TODO: randomize
        self.state = np.array([0, 0, 0,      # position
                              0, 0, 0, 1,    # orientation
                              0, 0, 0,       # linear velocity
                              0, 0, 0,       # angular velocity
                              0, 0, 0,       # linear acceleration
                              0, 0, 0,       # angular acceleration
                              545, 545, 545, # fan speeds
                              0.8, -1.25, 0.8, -1.25, 0.8, -1.25]) # nozzle angles
        
        obs = self.state[7:13] # lin_vel and ang_vel, TODO append gravity vector in body frame
        self.step_counter = 0
        info = {"state": self.state}
        return obs, info
    
    # First order actuator models, k[Hz] = 1 / tau[s]
    # tau[s] is the empirical time constant of the actuator state following a setpoint [s]
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
        # In this case, using the commanded PWM signal, so -1050 is already subtracted:
        fanspeeds_squared = np.square(fanspeeds)[:, np.newaxis]
        thrust_vectors = self.k_f * fanspeeds_squared * thrustdirections(nozzles_angles)
        # print(f"\nk_f: {self.k_f} \nomega_squared: \n{fanspeeds_squared}, \nthrustdirections(phi_state): \n{thrustdirections(phi_state)} \n thrust_vectors: \n{thrust_vectors}")
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

        # Update actuators
        nozzles_angles, fanspeeds = self.first_order_actuator_models(action)

        # Compute thrust vectors from EDF and servo states
        thrust_vectors = self.compute_thrust_vectors(nozzles_angles, fanspeeds)
        
        # Compute net force and torque
        force, torque = self.compute_forces_and_torques(thrust_vectors)
        
        # Extract current state
        position = self.state[0:3]
        orientation = R.from_quat(self.state[3:7])
        print(f"q: {self.state[3:7]}, orientation: {orientation.as_euler('xyz')}")
        lin_vel = self.state[7:10]
        ang_vel = self.state[10:13]
        
        print(f"torque: {torque}")

        # Update angular velocity and orientation
        ang_acc = torque / self.inertia # TODO np.cross(ang_vel, self.inertia @ angular_velocity)
        ang_vel += ang_acc * self.dt
        orientation *= R.from_rotvec(ang_vel * self.dt) # TODO calculate quaternion orientation from ang_vel
        print(f"q: {orientation.as_quat()}, orientation: {orientation.as_euler('xyz')}")
        
        # Update linear velocity and position
        # TODO calculate gravity vector in body frame
        g_body_frame = orientation.inv().apply(self.g)
        print(f"g_body: {g_body_frame}")
        lin_acc = force / self.mass - g_body_frame # TODO NE eqt. np.cross(ang_vel, lin_vel)
        lin_vel += lin_acc * self.dt
        position += orientation.apply(lin_vel) * self.dt
        
        # Update state
        self.state = np.concatenate([
            position,
            orientation.as_quat(),
            lin_vel,
            ang_vel,
            fanspeeds,
            nozzles_angles,
            lin_acc,
            ang_acc
        ])

        obs = np.concatenate([self.state[7:13], g_body_frame])

        # Compute reward
        velocity_penalty = np.linalg.norm(lin_vel) + np.linalg.norm(ang_vel)
        
        reward = -0.001*velocity_penalty
        
        # Check if truncated
        self.step_counter += 1
        if (self.step_counter > 1000):
            truncated = True

        info = {"state": self.state}
        
        return obs, reward, terminated, truncated, info
    
def plot_info(observations, infos, actions, rewards):

    observations = np.array(observations)
    actions = np.array(actions)
    states = np.array([info['state'] for info in infos])
    rewards = np.array(rewards)

    fig, axs = plt.subplots(5, 2, figsize=(20, 12))

    for i in range(0, 3, 1):
        axs[0,0].plot(states[:,i], label=f"position {i}")
        axs[0,0].legend()

    q = states[3:7]
    rpy = R.from_quat(q).as_euler('xyz', degrees=True)
    for i in range(3, 6, 1):
        axs[0,1].plot(rpy[0], label=f"roll [°]")
        axs[0,1].plot(rpy[1], label=f"pitch [°]")
        axs[0,1].plot(rpy[2], label=f"yaw [°]")
        axs[0,1].legend()

    for i in range(7, 10, 1):
        axs[1,0].plot(states[:,i], label=f"lin_vel {i}")
        axs[1,0].legend()

    for i in range(10, 13, 1):
        axs[1,1].plot(states[:,i], label=f"ang_vel {i}")
        axs[1,1].legend()

    for i in range(13, 16, 1):
        axs[2,0].plot(states[:,i], label=f"lin_acc {i}")
        axs[2,0].legend()

    for i in range(16, 19, 1):
        axs[2,1].plot(states[:,i], label=f"ang_acc {i}")
        axs[2,1].legend()

    for i in range(19, 22, 1):
        axs[3,0].plot(states[:,i], label=f"fan_speeds {i}")
        axs[3,0].plot(actions[:,i-12], label=f"fan_speeds actions{i}")
        axs[3,0].legend()

    for i in range(22, 28, 1):
        axs[3,1].plot(states[:,i], label=f"nozzle_angles {i}")
        axs[3,1].plot(actions[:,i-12], label=f"nozzle_angles actions{i}", linestyle='--')
        axs[3,1].legend()
    
    axs[4,0].plot(rewards, label="Reward", marker='.', linestyle='', markersize=3)
    axs[4,0].legend()

    plt.tight_layout()
    plt.show()

def train_MAV():

    env = MavEnv()

    model = PPO.load("ppo_mav_model", env=env)

    # Uncomment for new model
    # model = PPO("MlpPolicy", env, verbose=1)

    # eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/", log_path="./logs/", eval_freq=1000, deterministic=True, render=False)

    model.learn(total_timesteps=1_000_000)

    model.save("ppo_mav_model")


def evaluate_model():
    env = MavEnv()
    
    model = PPO.load("ppo_mav_model")

    obs, info = env.reset()
    # Record states and actions
    observations = [obs]
    infos = [info]
    actions = []
    rewards = []
    
    for i in range(1000):

        action, _states = model.predict(obs, deterministic=True)

        obs, reward, done, done, info = env.step(action)

        observations.append(obs)
        infos.append(info)
        actions.append(action)
        rewards.append(reward)
    
    plot_info(observations, infos, actions, rewards)
    

if __name__ == "__main__":

    print(f"test_MAV")
    # train_MAV()
    evaluate_model()
    