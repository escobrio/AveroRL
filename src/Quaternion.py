# SciPy uses xyzw quaternion convention, Robot Dynamics lecture wxyz
from scipy.spatial.transform import Rotation as R
import numpy as np
import time

def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    # Hamilton Quaternion Product. Source: Wikipedia Quaternion. Rearranged to return xyzw quaternion
    q3 = [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]
    return q3

def quaternion_rotate_vector(q, v):
    vector_quat = list(v) + [0]
    q_conjugate = [-q[0], -q[1], -q[2], q[3]]
    temp_result = quaternion_multiply(q, vector_quat)
    rotated_vector_quat = quaternion_multiply(temp_result, q_conjugate)
    return np.array(rotated_vector_quat[:3])

def quaternion_conjugate(q):
    return [-q[0], -q[1], -q[2], q[3]]


# Eqt. 2.97 From Robot Dynamics Script
def inverse_mapping_matrix(q: np.ndarray) -> np.ndarray:
    # Given a quaternion 'q' in [w, x, y, z] convention (As in Robot Dynamics lecture),
    # return the inverse of the mapping matrix E_R to be multiplied with ang_vel
    E_inv = 0.5 * np.array([
        [-q[1], -q[2], -q[3]],
        [ q[0],  q[3], -q[2]],
        [-q[3],  q[0],  q[1]],
        [ q[2], -q[1],  q[0]]
    ])
    return E_inv

def test_gravity_vector():
    print(f"\ntest_gravity_vector:")
    q1_body2world = np.array([0.382, 0, 0, 0.924])        # Roll 45°. Orientation of body frame in world frame
    g = np.array([0, 0, -9.81])                           # Gravity vector in world frame

    start = time.time()
    q1_world2body = quaternion_conjugate(q1_body2world)   # Conjugate to calculate world frame vector in body frame
    g_body = quaternion_rotate_vector(q1_world2body, g)   # Calculate gravity vector in body frame
    end = time.time()
    print(f"rotated vector: {g_body}, time: {end - start}")

    rot = R.from_quat(q1_body2world)

    start = time.time()
    g_body = rot.inv().apply(g)
    end = time.time()
    print(f"rotated vector: {g_body}, time: {end - start}")
    print(f"Roll, Pitch, Yaw [°] of bodyframe: {rot.as_euler('xyz', degrees=True)}")


# Compare integrating angular velocity to quaternions with pure quaternion math from robot dynamics lecture vs. SciPy Rotation object
def compare_quaternion_vs_scipy():
    print(f"\ncompare_quaternion_vs_scipy:")

    # Insert your parameters here:
    dT = 0.01
    time_steps = 10_000
    ang_vel = np.array([0, 0, 2*np.pi])
    q = [1, 0, 0, 0]  # [w, x, y, z]
    print(f"Integrating {time_steps * dT} seconds of and ang_vel = {ang_vel} into orientation quaternion using (1.) Robot Dynamics eqts. and (2.) SciPy Rotation object:")

    # (1.) Robot Dynamics eqt:
    start_time = time.time()
    for i in range(time_steps):
        E_R_inv = inverse_mapping_matrix(q)
        q_dot = E_R_inv @ ang_vel
        # Integration step
        q += q_dot * dT
        # Normalize
        q = q / np.sqrt(np.sum(q**2))
    end_time = time.time()

    q = [q[1], q[2], q[3], q[0]] # Rearrange wxyz quaternion to xyzw quaternion which is used by scipy
    rot = R.from_quat(q)
    print(f"\n(1.) time to calculate: {end_time - start_time:.4f}s")
    print(f"Quaternion q: \n{rot.as_quat()}")
    print(f"In rpy: \n{rot.as_euler('xyz', degrees=True)}")

    # (2.) SciPy Rotation object:
    q = [0, 0, 0, 1]    # [x, y, z, w]
    rot = R.from_quat(q)

    start_time = time.time()
    for i in range(time_steps):
        rot = rot * R.from_rotvec(np.array(ang_vel) * dT)
    end_time = time.time()

    q = rot.as_quat()

    print(f"\n(2.) time to calculate: {end_time - start_time:.4f}s")
    print(f"Quaternion q: \n{rot.as_quat()}")
    print(f"In rpy: \n{rot.as_euler('xyz', degrees=True)}")

    print(f"\nConclusion: Both ways to integrate ang_vel to orientation are ~equally fast, \nSciPy Rotation object seems to be more accurate and easier to work with and interpret")


if __name__ == "__main__":
    
    # Choose which one to test
    # test_gravity_vector()
    compare_quaternion_vs_scipy()
