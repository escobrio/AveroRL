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


if __name__ == "__main__":
    
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
