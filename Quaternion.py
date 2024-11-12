from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

# Quaternion convention is xyzw
r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])

print(f"Unit Quaternion: \n{r.as_quat()}")
print(f"Rotation Vec [rad]: \n{r.as_rotvec()}")
print(f"Rotation matrix: \n{r.as_matrix()}")
print(f"roll, pitch, yaw [°]: \n{r.as_euler('xyz', degrees=True)}")

print(f"Inverse [rpy]: \n{r.inv().as_euler('xyz', degrees=True)}")

g = np.array([0, 0, -9.81])
print(f"Apply rotation to gravity vector: r * g = \n{r.apply(g)}")
