### From mav_avero/nodes/avero_ctrl/src/allocation_python_new/helper_functions_symb_new_sim.py
import numpy as np

# Normal vector of the thrustvector exiting out of the nozzle 1 depending on the swivel nozzle angles phi11, phi12
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

# Distance from Base Frame to Endeffector of Nozzle 1 depending on the swivel nozzle angles phi11 and phi12
def r_BE_1(phi1, phi2):
    return np.array([
        0.0203 * np.sin(phi1) * np.cos(phi2) + 0.0779 * np.sin(phi1) + 0.0288 * np.sin(phi2) * np.cos(phi1) + 0.1507,
        -0.0098 * np.sin(phi1) * np.sin(phi2) + 0.0070 * np.cos(phi1) * np.cos(phi2) + 0.0266 * np.cos(phi1) + 0.0191 * np.cos(phi2) - 0.2290,
        0.0270 * np.sin(phi1) * np.sin(phi2) - 0.0191 * np.cos(phi1) * np.cos(phi2) - 0.0732 * np.cos(phi1) + 0.0070 * np.cos(phi2) - 0.0405
    ])

def r_BE_2(phi1, phi2):
    return np.array([
        0.0085 * np.sin(phi1) * np.sin(phi2) - 0.0102 * np.sin(phi1) * np.cos(phi2) - 0.0389 * np.sin(phi1) 
        - 0.0144 * np.sin(phi2) * np.cos(phi1) - 0.0060 * np.cos(phi1) * np.cos(phi2) - 0.0231 * np.cos(phi1) 
        - 0.0166 * np.cos(phi2) + 0.1271,
        
        0.0049 * np.sin(phi1) * np.sin(phi2) + 0.0176 * np.sin(phi1) * np.cos(phi2) + 0.0674 * np.sin(phi1) 
        + 0.0249 * np.sin(phi2) * np.cos(phi1) - 0.0035 * np.cos(phi1) * np.cos(phi2) - 0.0133 * np.cos(phi1) 
        - 0.0096 * np.cos(phi2) + 0.2446,
        
        0.0270 * np.sin(phi1) * np.sin(phi2) - 0.0191 * np.cos(phi1) * np.cos(phi2) - 0.0732 * np.cos(phi1) 
        + 0.0070 * np.cos(phi2) - 0.0405
    ])

def r_BE_3(phi1, phi2):
    return np.array([
        -0.0085 * np.sin(phi1) * np.sin(phi2) - 0.0102 * np.sin(phi1) * np.cos(phi2) - 0.0389 * np.sin(phi1) 
        - 0.0144 * np.sin(phi2) * np.cos(phi1) + 0.0060 * np.cos(phi1) * np.cos(phi2) + 0.0231 * np.cos(phi1) 
        + 0.0166 * np.cos(phi2) - 0.2682,
        
        0.0049 * np.sin(phi1) * np.sin(phi2) - 0.0176 * np.sin(phi1) * np.cos(phi2) - 0.0674 * np.sin(phi1) 
        - 0.0249 * np.sin(phi2) * np.cos(phi1) - 0.0035 * np.cos(phi1) * np.cos(phi2) - 0.0133 * np.cos(phi1) 
        - 0.0096 * np.cos(phi2) - 0.0117,
        
        0.0270 * np.sin(phi1) * np.sin(phi2) - 0.0191 * np.cos(phi1) * np.cos(phi2) - 0.0732 * np.cos(phi1) 
        + 0.0070 * np.cos(phi2) - 0.0405
    ])

def r_BE(phi_vector):
  phi11, phi12, phi21, phi22, phi31, phi32 = phi_vector
  return r_BE_1(phi11, phi12), r_BE_2(phi21, phi22), r_BE_3(phi31, phi32)


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
  states_phi = np.array([np.pi, 0, np.pi, 0, np.pi, 0])
  print(f"\nThrust normal vectors for nozzle angles: {states_phi}: \n{thrustdirections(states_phi)}")

  # Set three fan speeds omega
  states_omega = np.array([900, 800, 700])
  omega_squared = np.square(states_omega)[:, np.newaxis]
  k_f = 0.00006
  # Calculate thrust vectors
  thrust_vectors = k_f * omega_squared * thrustdirections(states_phi)
  print(f"\n Thrust vector for fan speeds: {states_omega}, nozzle angles: {states_phi} and k_f = {k_f} :\n{thrust_vectors}")

  r_BE_1, r_BE_2, r_BE_3 = r_BE(states_phi)
  print(f"\nr_BE_1, r_BE_2, r_BE_3 = \n{r_BE(states_phi)} ")
      
if __name__ == "__main__":
  main()
  