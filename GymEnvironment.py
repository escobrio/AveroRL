import numpy as np
from scipy.spatial.transform import Rotation
import gymnasium as gym
from ForwardKinematics import thrustdirections, r_BE

class MavEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # State: [x, y, z, roll, pitch, yaw, dx, dy, dz, droll, dpitch, dyaw]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf] * 12),
            high=np.array([np.inf] * 12),
            dtype=np.float32
        )
        
        # Actions: [EDF1, EDF2, EDF3, servo1, servo2, servo3, servo4, servo5, servo6]
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0, 0, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2]),
            high=np.array([1, 1, 1, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2]),
            dtype=np.float32
        )
        
        # Physical parameters
        self.mass = 1.0  # kg
        self.inertia = np.array([0.1, 0.1, 0.1])  # kg*m^2
        self.arm_length = 0.2  # m
        self.max_thrust = 10.0  # N
        self.dt = 0.01  # s
        self.g = 9.81  # m/s^2
        self.k_f = 0.00006 # Thrust constant, Thrust_force = k_f * omega²

        # EDF positions (120 degrees apart)
        self.edf_positions = np.array([
            [self.arm_length * np.cos(0), self.arm_length * np.sin(0), 0],
            [self.arm_length * np.cos(2*np.pi/3), self.arm_length * np.sin(2*np.pi/3), 0],
            [self.arm_length * np.cos(4*np.pi/3), self.arm_length * np.sin(4*np.pi/3), 0]
        ])
        
        # Initialize visualization
        self.viewer = None
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        # Initialize state: slight offset from zero to make it interesting
        self.state = np.array([0, 0, 1,  # position
                              0.1, 0.1, 0,  # orientation
                              0, 0, 0,  # linear velocity
                              0, 0, 0])  # angular velocity
        return self.state, {}
    
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
        force[2] -= self.mass * self.g  # Add gravity
        
        # Compute torques from each thrust
        torque = np.zeros(3)
        action_phi = np.zeros(6)
        r_BE_1, r_BE_2, r_BE_3 = r_BE(action_phi)
        torque += np.cross(r_BE_1, thrust_vectors[0])
        torque += np.cross(r_BE_2, thrust_vectors[1])
        torque += np.cross(r_BE_3, thrust_vectors[2])
        
        return force, torque
    
    def step(self, action):
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
        linear_acc = force / self.mass
        linear_vel += linear_acc * self.dt
        position += linear_vel * self.dt
        
        # Update angular velocity and orientation
        angular_acc = torque / self.inertia
        angular_vel += angular_acc * self.dt
        orientation += angular_vel * self.dt
        
        # Update state
        self.state = np.concatenate([
            position,
            orientation,
            linear_vel,
            angular_vel
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
    
    # def render(self):

def test_MAV():
    env = MavEnv()
    state = env.reset()
    
    for _ in range(10):
        # Test with simple hover action
        action = np.array([0.5, 0.5, 0.5,  # EDF powers
                          0, 0,             # EDF1 angles
                          0, 0,             # EDF2 angles
                          0, 0])            # EDF3 angles
        
        state, reward, done, _, _, force, torque = env.step(action)
        print(f"force: {force}, torque: {torque}")
        # env.render()

if __name__ == "__main__":
    print(f"test_MAV")
    test_MAV()