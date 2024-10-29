### From mav_avero/nodes/avero_ctrl/src/allocation_python_new/helper_functions_symb_new_sim.py
import numpy as np


def thrustvector_nozzle1_bodyframe(phi1, phi2):
  thrust_vector = np.array([
    [-0.5 * np.sin(phi1) * np.cos(phi2) - 0.5 * np.sin(phi1) - 0.7071 * np.sin(phi2) * np.cos(phi1)],
    [0.24183 * np.sin(phi1) * np.sin(phi2) - 0.171 * np.cos(phi1) * np.cos(phi2) - 0.171 * np.cos(phi1) - 0.46985 * np.cos(phi2) + 0.46985],
    [-0.66447 * np.sin(phi1) * np.sin(phi2) + 0.46985 * np.cos(phi1) * np.cos(phi2) + 0.46985 * np.cos(phi1) - 0.171 * np.cos(phi2) + 0.171]
  ])
  return thrust_vector



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

print(thrustvector_nozzle1_bodyframe(2*np.pi,0))
