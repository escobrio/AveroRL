import numpy as np
from scipy.spatial.transform import Rotation
import gymnasium as gym
from ForwardKinematics import thrustdirections, r_BE
import matplotlib.pyplot as plt

class MavEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Every Gym Environment must have observation_space
        # State: [x, y, z, roll, pitch, yaw, dx, dy, dz, droll, dpitch, dyaw]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * 12),
            high=np.array([np.inf] * 12),
            dtype=np.float32
        )
        
        # Every Gym Environment must have action_space
        # Actions: [EDF1_des, EDF2_des, EDF3_des, servo1_des, servo2_des, servo3_des, servo4_des, servo5_des, servo6_des]
        self.action_space = gym.spaces.Box(
            low=np.array([1050, 1050, 1050, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi]),
            high=np.array([1950, 1950, 1950, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi]),
            dtype=np.float32
        )

        # 21 states
        self.state = np.array([0, 0, 1,  # [:3] position
                              0, 0, 0,  # [3:6] orientation
                              0, 0, 0,  # [6:9] linear velocity
                              0, 0, 0,  # [9:12]angular velocity
                              1050, 1050, 1050, # [12:15] fan speeds
                              0, 0, 0, 0, 0, 0]) # [15:21] nozzle angles
        
        # Physical parameters
        self.mass = 5.218  # kg
        self.inertia = np.array([0.059829689, 0.06088309, 0.098981953])  # kg*m^2
        self.dt = 0.01  # s
        self.g = np.array([0, 0, 9.81])  # m/s^2
        self.k_f = 0.00006 # Thrust constant, Thrust_force = k_f * omega²
        self.k_phi = 6 # Hz, First order nozzle angle model, 1/tau where tau is time constant
        self.k_omega = 12 # Hz, First order fan speed model TODO this is actually k_forceomega
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Initialize state: 
        # TODO: randomize
        self.state = np.array([0, 0.1, 1,  # position
                              0, 0.1, 0.2,  # orientation
                              0, 0.1, 0.2,  # linear velocity
                              0, 0.1, 0.2,  # angular velocity
                              1050, 1150, 1250, # fan speeds
                              -2, -1, 0, 1, 2, 3]) # nozzle angles
        return self.state
    
    def first_order_actuator_models(self, action):
        phi_des = action[3:]
        phi_state = self.state[15:]
        phi_dot = self.k_phi * (phi_des - phi_state)
        phi_state += phi_dot * self.dt

        omega_des = action[:3]
        omega_state = self.state[12:15]
        omega_dot = self.k_omega * (omega_des - omega_state)
        omega_state += omega_dot * self.dt

        return phi_state, omega_state

    def compute_thrust_vectors(self, action):
        # """Compute thrust vectors for each EDF based on servo angles."""
        # thrusts = action[:3] * self.max_thrust
        # servo_angles = action[3:]  # 2 angles per EDF

        
        action_omega = np.array([900, 800, 700])
        action_omega_squared = np.square(action_omega)[:, np.newaxis]
        action_phi = np.zeros(6)
        thrust_vectors = self.k_f * action_omega_squared * thrustdirections(action_phi)
        return thrust_vectors
    
    def compute_forces_and_torques(self, thrust_vectors):
        """Compute net force and torque from thrust vectors."""
        # Net force is sum of all thrust vectors
        force = np.sum(thrust_vectors, axis=0)
        
        # Compute torques from each thrust
        torque = np.zeros(3)
        action_phi = np.zeros(6)
        r_BE_1, r_BE_2, r_BE_3 = r_BE(action_phi)
        torque += np.cross(r_BE_1, thrust_vectors[0])
        torque += np.cross(r_BE_2, thrust_vectors[1])
        torque += np.cross(r_BE_3, thrust_vectors[2])
        
        return force, torque
    
    def step(self, action):
        # Update actuators
        phi_state, omega_state = self.first_order_actuator_models(action)

        # Get thrust vectors from EDF and servo settings
        thrust_vectors = self.compute_thrust_vectors(action)
        
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
        linear_acc = force / self.mass - self.g # TODO NE eqt. np.cross(angular_vel, linear_vel)
        linear_vel += linear_acc * self.dt
        position += linear_vel * self.dt
        
        # Update angular velocity and orientation
        angular_acc = torque / self.inertia # TODO np.cross(angular_vel, self.inertia @ angular_velocity)
        angular_vel += angular_acc * self.dt
        orientation += angular_vel * self.dt
        
        # Update state
        self.state = np.concatenate([
            position,
            orientation,
            linear_vel,
            angular_vel,
            omega_state,
            phi_state
        ])
        
        # Compute reward
        target_position = np.array([0, 0, 1])  # Hover at 1m
        target_orientation = np.zeros(3)        # Level orientation
        
        position_error = np.linalg.norm(position - target_position)
        orientation_error = np.linalg.norm(orientation - target_orientation)
        velocity_penalty = np.linalg.norm(linear_vel) + np.linalg.norm(angular_vel)
        
        reward = -(position_error + 0.5 * orientation_error + 0.1 * velocity_penalty)
        
        # Check if done
        done = position_error < 0.1 and orientation_error < 0.1 and velocity_penalty < 0.1
        
        return self.state, reward, done, False, {}, force, torque
    
    def plot_states(self, states, actions):

        print(f"states: {len(states[0])}")
        states = np.array(states)
        actions = np.array(actions)
        print(states.shape)

        fig, axs = plt.subplots(3, 2, figsize=(12, 10))

        for i in range(0, 3, 1):
            axs[0,0].plot(states[:,i], label=f"position {i}")
            axs[0,0].legend()

        for i in range(3, 6, 1):
            axs[0,1].plot(states[:,i], label=f"orientation {i}")
            axs[0,1].legend()

        for i in range(6, 9, 1):
            axs[1,0].plot(states[:,i], label=f"linear velocity {i}")
            axs[1,0].legend()

        for i in range(9, 12, 1):
            axs[1,1].plot(states[:,i], label=f"angular velocity {i}")
            axs[1,1].legend()

        for i in range(12, 15, 1):
            axs[2,0].plot(states[:,i], label=f"fan speeds {i}")
            axs[2,0].plot(actions[:,i-12], label=f"fan speeds {i}", linestyle='--')
            axs[2,0].legend()

        for i in range(15, 21, 1):
            axs[2,1].plot(states[:,i], label=f"nozzle angle {i}")
            axs[2,1].plot(actions[:,i-12], label=f"nozzle angle desired {i}", linestyle='--')
            axs[2,1].legend()
        
        plt.show()


def test_MAV():
    env = MavEnv()
    state = env.reset()
    
    # Record states and actions
    states = [state]
    actions = []
    
    for _ in range(100):
        # Test with simple hover action
        action = np.array([1500, 1500, 1500,  # EDF powers
                          0, 0,             # EDF1 angles
                          0, 0,             # EDF2 angles
                          0, 0])            # EDF3 angles
        
        state, reward, done, _, _, force, torque = env.step(action)
        states.append(state)
        actions.append(action)

    env.plot_states(states, actions)    


if __name__ == "__main__":
    print(f"test_MAV")
    test_MAV()
    