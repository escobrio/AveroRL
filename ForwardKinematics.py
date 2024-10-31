### From mav_avero/nodes/avero_ctrl/src/allocation_python_new/helper_functions_symb_new_sim.py
import numpy as np

# Normal vectorr of the thrustvector exiting out of the nozzle 1 depending on the swivel nozzle angles phi11, phi12
def thrustdirection_nozzle1_bodyframe(phi1, phi2):
  thrust_vector = np.array([
    -0.5 * np.sin(phi1) * np.cos(phi2) - 0.5 * np.sin(phi1) - 0.7071 * np.sin(phi2) * np.cos(phi1),
    0.24183 * np.sin(phi1) * np.sin(phi2) - 0.171 * np.cos(phi1) * np.cos(phi2) - 0.171 * np.cos(phi1) - 0.46985 * np.cos(phi2) + 0.46985,
    -0.66447 * np.sin(phi1) * np.sin(phi2) + 0.46985 * np.cos(phi1) * np.cos(phi2) + 0.46985 * np.cos(phi1) - 0.171 * np.cos(phi2) + 0.171
  ])
  return thrust_vector

def thrustdirection_nozzle2_bodyframe(phi1, phi2):
  thrust_vector = np.array([
    -0.209445 * np.sin(phi1) * np.sin(phi2) + 0.25 * np.sin(phi1) * np.cos(phi2) + 0.25 * np.sin(phi1) + 0.35355 * np.sin(phi2) * np.cos(phi1) + 0.1481 * np.cos(phi1) * np.cos(phi2) + 0.1481 * np.cos(phi1) + 0.4069 * np.cos(phi2) - 0.4069,
    -0.12092 * np.sin(phi1) * np.sin(phi2) - 0.433 * np.sin(phi1) * np.cos(phi2) -0.433 * np.sin(phi1) - 0.61235 * np.sin(phi2) * np.cos(phi1) + 0.0855 * np.cos(phi1) * np.cos(phi2) + 0.0855 * np.cos(phi1) + 0.2349 * np.cos(phi2) - 0.2349,
    -0.66447 * np.sin(phi1) * np.sin(phi2) + 0.46985 * np.cos(phi1) * np.cos(phi2) + 0.46985 * np.cos(phi1) - 0.171 * np.cos(phi2) + 0.171
  ])
  return thrust_vector

def thrustdirection_nozzle3_bodyframe(phi1, phi2):
  thrust_vector = np.array([
    0.20945 * np.sin(phi1) * np.sin(phi2) + 0.25 * np.sin(phi1) * np.cos(phi2) + 0.25 * np.sin(phi1) + 0.35355 * np.sin(phi2) * np.cos(phi1) - 0.1481 * np.cos(phi1) * np.cos(phi2) - 0.1481 * np.cos(phi1) - 0.4069 * np.cos(phi2) + 0.4069,
    -0.1209 * np.sin(phi1) * np.sin(phi2) + 0.433 * np.sin(phi1) * np.cos(phi2) + 0.433 * np.sin(phi1) + 0.61235 * np.sin(phi2) * np.cos(phi1) + 0.0855 * np.cos(phi1) * np.cos(phi2) + 0.0855 * np.cos(phi1) + 0.2349 * np.cos(phi2) - 0.2349,
    -0.66447 * np.sin(phi1) * np.sin(phi2) + 0.46985 * np.cos(phi1) * np.cos(phi2) + 0.46985 * np.cos(phi1) - 0.171 * np.cos(phi2) + 0.171
  ])
  return thrust_vector

def thrustdirections(phi_vector):
  phi11, phi12, phi21, phi22, phi31, phi32 = phi_vector
  return np.array([
    thrustdirection_nozzle1_bodyframe(phi11, phi12),
    thrustdirection_nozzle2_bodyframe(phi21, phi22),
    thrustdirection_nozzle3_bodyframe(phi31, phi32)
  ])

# Homogeneous Transformation matrices from MAV's body frame to base nozzle frame
# Contains 3x3 rotation matrix R and 3x1 translation vector t
T_body_to_nozzle1_base = np.array([
  [ 0.0,     0.0,    -1.0,  0.15],
  [ 0.3420, -0.9397,  0.0, -0.117],
  [-0.9397, -0.342,   0.0,  0.0],
  [ 0.0,     0.0,     0.0,  1.0]
])

T_body_to_nozzle2_base = np.array([
  [-0.2962, -0.8138,  0.5,    0.03067],
  [-0.1710,  0.4698, -0.8660, 0.18895],
  [-0.9397, -0.3420,  0.0,    0.0],
  [ 0.0,     0.0,     0.0,    1.0]
])

T_body_to_nozzle3_base = np.array([
  [ 0.2962, -0.8138, 0.5,    -0.1717],
  [-0.1710,  0.4698, 0.8660, -0.06737],
  [-0.9397, -0.3420, 0.0,     0.0],
  [ 0.0,     0.0,    0.0,     0.0]
])


def main():
  # Example usage:
  # Set the six nozzle angles
  states_phi = np.zeros(6)
  print(f"\nThrust normal vectors for nozzle angles: {states_phi}: \n{thrustdirections(states_phi)}")

  # Set three fan speeds omega
  action_omega = np.array([900, 800, 700])
  omega_squared = np.square(action_omega)[:, np.newaxis]
  action_phi = 0 * np.ones(6)
  k_f = 0.00006
  # Calculate thrust vectors
  thrust_vectors = k_f * omega_squared * thrustdirections(action_phi)
  print(f"\n Thrust vector for fan speeds: {action_omega}, nozzle angles: {action_phi} and k_f = {k_f} :\n{thrust_vectors}")
      
if __name__ == "__main__":
  main()
  