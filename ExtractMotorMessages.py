import rosbag
import numpy as np
import matplotlib.pyplot as plt

# Extracts commanded EDF fan speeds and nozzle angles
def extract_motor_messages(bagfile_path):
    motor_messages = []
    edf_messages = []
    with rosbag.Bag(bagfile_path, 'r') as bag:

        # Nozzle angle commands 
        for topic, msg, t in bag.read_messages(topics=['/command/u']):
            edf_message = []
            for i in range(3):
                edf_message.append(msg.data[i])
            
            edf_messages.append(np.array(edf_message))

        # EDF fan speed commands
            motor_message = []
            for i in range(3,9):
                motor_message.append(msg.data[i])
            
            motor_messages.append(np.array(motor_message))
    
    return motor_messages, edf_messages

# Extracts commanded EDF fan speeds and nozzle angles
def extract_pose(bagfile_path):
    positions = []
    orientations = []
    with rosbag.Bag(bagfile_path, 'r') as bag:

        # Nozzle angle commands 
        for topic, msg, t in bag.read_messages(topics=['/Avero_Dove/msf_core/pose']):
            position = []
            orientation = []
            position.append(msg.pose.pose.position.x)
            position.append(msg.pose.pose.position.y)
            position.append(msg.pose.pose.position.z)
            positions.append(position)
            orientation.append(msg.pose.pose.orientation.x)
            orientation.append(msg.pose.pose.orientation.y)
            orientation.append(msg.pose.pose.orientation.z)
            orientation.append(msg.pose.pose.orientation.w)
            orientations.append(orientation)
                
    return positions, orientations

# Plot Actuator Commands
def plot_actuator_cmds(motor_msgs, edf_msgs):
    
    motor_msgs = np.array(motor_msgs)
    edf_msgs = np.array(edf_msgs)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,12))

    ax1.plot(motor_msgs[:, 0], linestyle='None', marker='.', markersize=3, label='phi_11')
    ax1.plot(motor_msgs, alpha=0.1)
    ax1.set_title("Commanded Nozzle Angles (Not Servo Positions!)")
    ax1.set_xlabel("Time Step at 60Hz")
    ax1.set_ylabel("Nozzle Angle [rad]")
    ax1.legend()

    ax2.plot(edf_msgs[:, 0], linestyle='None', marker='.', markersize=3, label='omega_1')
    ax2.plot(edf_msgs[:, 1], linestyle='None', marker='.', markersize=3, label='omega_2')
    ax2.plot(edf_msgs[:, 2], linestyle='None', marker='.', markersize=3, label='omega_3')
    ax2.plot(edf_msgs, alpha=0.1)
    ax2.set_title("Commanded EDF PWM")
    ax2.set_xlabel("Time Step at 60Hz")
    ax2.set_ylabel("(PWM-1050)")
    ax2.legend()

    plt.show()

# Plot pose Signal
def plot_pose(positions, orientations):
    
    positions = np.array(positions)
    orientations = np.array(orientations)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,12))

    ax1.plot(positions[:, 0], linestyle='None', marker='.', markersize=3, label='pos_x')
    ax1.plot(positions[:, 1], linestyle='None', marker='.', markersize=3, label='pos_y')
    ax1.plot(positions[:, 2], linestyle='None', marker='.', markersize=3, label='pos_z')
    ax1.plot(positions, alpha=0.1)
    ax1.set_title("Position xyz")
    ax1.set_xlabel("Time Step at 190 Hz")
    ax1.set_ylabel("Position [m]")
    ax1.legend()

    ax2.plot(orientations[:, 0], linestyle='None', marker='.', markersize=3, label='q_x')
    ax2.plot(orientations[:, 1], linestyle='None', marker='.', markersize=3, label='q_y')
    ax2.plot(orientations[:, 2], linestyle='None', marker='.', markersize=3, label='q_z')
    ax2.plot(orientations[:, 3], linestyle='None', marker='.', markersize=3, label='q_w')
    ax2.plot(orientations, alpha=0.1)
    ax2.set_title("Orientation Quaternion xyzw")
    ax2.set_xlabel("Time Step at 190Hz")
    ax2.set_ylabel("Quaternion value xyzw")
    ax2.legend()
    
    plt.show()


if __name__ == "__main__":

    # Insert desired filepath here!
    bagfile_path = '/home/dedi/BachelorThesis/AveroRL/T10-cut.bag'
    
    motor_msgs, edf_msgs = extract_motor_messages(bagfile_path)
    positions, orientations = extract_pose(bagfile_path)

    # np.save("sample_commands_edf.npy", edf_msgs, allow_pickle=True)
    # np.save("sample_commands_nozzle.npy", motor_msgs, allow_pickle=True)
    np.save("sample_commands_positions.npy", positions, allow_pickle=True)
    np.save("sample_commands_orientations.npy", orientations, allow_pickle=True)

    plot_actuator_cmds(motor_msgs, edf_msgs)
    plot_pose(positions, orientations)
