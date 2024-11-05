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


if __name__ == "__main__":

    # Insert desired filepath here!
    bagfile_path = '/home/dedi/BachelorThesis/AveroRL/T10-cut.bag'
    
    motor_msgs, edf_msgs = extract_motor_messages(bagfile_path)

    # Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,12))

    ax1.plot(motor_msgs, linestyle='None', marker='.', markersize=3)
    ax1.plot(motor_msgs, alpha=0.1)
    ax1.set_title("Commanded Nozzle Angles (Not Servo Positions!)")
    ax1.set_xlabel("Time Step at 60Hz")
    ax1.set_ylabel("Nozzle Angle [rad]")

    ax2.plot(edf_msgs, linestyle='None', marker='.', markersize=3)
    ax2.plot(edf_msgs, alpha=0.1)
    ax2.set_title("Commanded EDF PWM")
    ax2.set_xlabel("Time Step at 60Hz")
    ax2.set_ylabel("(PWM-1050)")
    
    plt.show()
