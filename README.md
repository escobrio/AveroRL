# AveroRL
"Reinforcement Learning Controller for the Omnidirectional MAV Avero" Bachelor Thesis

## Repository structure
- src/GymEnvironment.py is the main file, it contains a gymnasium environment of the Avero MAV to simulate the MAV's dynamics and to train and evaluate a flying policy with the the PPO reinforcement learning algorithm from stable_baselines3.
- src/ForwardKinematics.py is used to calculate thrustforce and torques depending on actuator inputs.
- src/Quaternion.py implements Quaternion math from Robot Dynamics lecture. It also contains a function to compare the performance to SciPy's rotation object.
- src/ExtractMotorMessages.py is a script to extract actuator commands (fanspeeds and nozzleangles) from a real in-flight rosbag.
- data/ppo_mav_model.zip is the trained RL model.

## Coordinate System Convention:
![CoordinateSystemAvero](https://github.com/user-attachments/assets/70200639-49b3-4ec6-b187-a7daf5bcbf9d)

## Weight and Inertia Tensor:
<img width="1230" alt="Inertia_MAV" src="https://github.com/user-attachments/assets/818b0ce2-0ae0-4d56-b450-d4c81a013875">
