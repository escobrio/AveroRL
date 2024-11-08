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
        return self.state
    
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

        states = np.array(states)
        actions = np.array(actions)

        fig, axs = plt.subplots(4, 2, figsize=(20, 12))

        duration = 80
        time_states = np.linspace(0, duration, len(states))

        axs[0,0].plot(time_states, states[:,0], label="pos_x_sim", c='C0')
        axs[0,0].plot(time_states, states[:,1], label="pos_y_sim", c='C1')
        axs[0,0].plot(time_states, states[:,2], label="pos_z_sim", c='C2')
        axs[0,0].legend()
        axs[0,0].set_xlabel("Time [s]")
        axs[0,0].set_ylabel("Position [m]")

        for i in range(3, 6, 1):
            axs[0,1].plot(states[:,i], label=f"orientation {i}")
            axs[0,1].legend()

        for i in range(6, 9, 1):
            axs[1,0].plot(states[:,i], label=f"linear velocity {i}")
            axs[1,0].legend()

        for i in range(9, 12, 1):
            axs[1,1].plot(states[:,i], label=f"angular velocity {i}")
            axs[1,1].legend()

        for i in range(21, 24, 1):
            axs[2,0].plot(states[:,i], label=f"linear acceleration {i}")
            axs[2,0].legend()

        for i in range(24, 27, 1):
            axs[2,1].plot(states[:,i], label=f"angular acceleration {i}")
            axs[2,1].legend()

        for i in range(12, 15, 1):
            axs[3,0].plot(states[:,i], label=f"fan speeds {i}")
            axs[3,0].plot(actions[:,i-12], label=f"fan speeds {i}", linestyle='--')
            axs[3,0].legend()

        for i in range(15, 21, 1):
            axs[3,1].plot(states[:,i], label=f"nozzle angle {i}")
            axs[3,1].plot(actions[:,i-12], label=f"nozzle angle desired {i}", linestyle='--')
            axs[3,1].legend()
        
        plt.tight_layout()
        plt.show()


def test_MAV(commands_edf, commands_nozzle):
    env = MavEnv()
    state = env.reset()
    
    # Record states and actions
    states = [state]
    actions = []
    forces = []
    
    for i in range(len(commands_edf)):
        # Test with actual actuator commands
        action = np.concatenate((commands_edf[i], commands_nozzle[i]))
        
        state, reward, done, _, _, force, torque = env.step(action)
        states.append(state)
        actions.append(action)
        forces.append(force)

    plt.plot(forces)
    plt.show()
    env.plot_states(states, actions)


if __name__ == "__main__":

    # Load sample commands to test out simulation
    commands_edf = 545 * np.ones((4509, 3))
    commands_nozzle = np.array([0.8, -1.25, 0.8, -1.25, 0.8, -1.25])
    commands_nozzle = np.tile(commands_nozzle, (4509, 1))

    print(f"test_MAV")
    test_MAV(commands_edf, commands_nozzle)
    