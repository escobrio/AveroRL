# AveroRL
"Reinforcement Learning Controller for the Omnidirectional MAV Avero" Bachelor Thesis

## Repository structure
- src/GymEnvironment.py is the main file, it contains a gymnasium environment of the Avero MAV to simulate the MAV's dynamics and to train and evaluate a flying policy with the the PPO reinforcement learning algorithm from stable_baselines3.
- src/ForwardKinematics.py is used to calculate thrustforce and torques depending on actuator inputs.
- src/Quaternion.py implements Quaternion math from Robot Dynamics lecture. It also contains a function to compare the performance to SciPy's rotation object.
- src/ExtractMotorMessages.py is a script to extract actuator commands (fanspeeds and nozzleangles) from a real in-flight rosbag.
- data/ppo_mav_model.zip is the trained RL model.

## The MAV
During the 1-year focus project AVERO in 2024 at the Autonomous Systems Lab at ETH Zürich, 8 Mechanical Engineering students developed an omnidirectional, safe-to-interact-with MAV. It has 9 actuators: 3 Electric Ducted Fans and 6 Servo Motors used to control Swivel Nozzles for thrust vectoring. Stable flight was acheived with a model-based PID Controller, as can be seen on the [official Avero Website](https://avero.ethz.ch/). This bachelor thesis wants to explore an alternative way to control this overactuated aerial robot using an Artificial Neural Network Policy trained with Reinforcement Learning algorithms in a simulated environment. The goal is to stably hover the MAV in all 6 DoF of space.

![Avero_Rendering_76](https://github.com/user-attachments/assets/c6b3e0f1-89f0-4f31-b037-c5699d83eb4a)


## Coordinate System Convention:
![CoordinateSystemAvero](https://github.com/user-attachments/assets/70200639-49b3-4ec6-b187-a7daf5bcbf9d)

## Weight and Inertia Tensor:
<img width="1230" alt="Inertia_MAV" src="https://github.com/user-attachments/assets/818b0ce2-0ae0-4d56-b450-d4c81a013875">
