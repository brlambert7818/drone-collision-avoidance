# Autonomous UAV Collision Avoidance via Simulated Collision

### Brian Lambert

University of Edinburgh MSc Cognitive Science dissertation on drone collision avoidance using reinforcement learning and collision heuristics. Unmanned aerial vehicles (UAV) have grown rapidly in popularity, especially bene-fiting recently from improvements in autonomous control provided by ReinforcementLearning  (RL).  However,  RL  has  failed  to  solve  the  critical  task  of  UAV  collisionavoidance.  This project aimed to resolve this issue by integrating a collision avoidanc eheuristic, which simulates a realistic collision, into an RL model trained to accomplish a higher-level task.  Two flight controllers were developed and compared:  A hybrid model in which a collisions heuristic was integrated into an RL agent pre-trained toonly accomplish a high-level task and RL agent trained to simultaneously avoid collisions and accomplish a task. We found that the hybrid RL-heuristic flight controller is superior to a solely RL model both in terms of task completion and collision avoidance

## Simulation Software Stack
- The simulation software stack is comprised of ROS, Gazebo, and OpenAI Gym
- This code was tested on `Ubuntu 16.04`
- The code relies on the `sim_cf` package from https://github.com/wuwushrek/sim_cf
  - Following the intall instructions in the above repo will install and compile ROS, Gazebeo7, and the Crazyflie Gazebo firmware
- The drone used for this research project is the `Crazyflie 2.0` from Bitcraze https://www.bitcraze.io/products/old-products/crazyflie-2-0/
![Crazyflie 2.0](/cf_real.jpg)
- This code relies on the PPO2 Reinforcement Learning algorithm provided by 'Stable Baselines' from https://github.com/hill-a/stable-baselines

## Training
- The default Gazebo world to train in is 'empty_word' which is a simple ground plane
  - This can be changed to 'box_world' by changing the default world name in `/sim_cf/crazyflie_gazebo/launch/crazyflie_sim.launch` for single drone use or         `/sim_cf/crazyflie_gazebo/launch/multiple_cf_sim.launch` for multi drone use
  ![Crazyflie 2.0](/empty_world.png 50%) ![Crazyflie 2.0](/box_world.png 50%)
- To begin trianing a single drone to nagivate towards a goal, execute the following terminal command: `roslaunch drone_training cf_train.launch`
- To begin trianing a single drone to nagivate towards a goal while avoidng obstacles, execute the following terminal command: `roslaunch drone_training cf_train_obstacles.launch`
- The trained models can be evualted with the following terminal command: `roslaunch drone_training cf_evaluate.launch`
  - In the file `cf_evaluate.py` you must specify where to locate the trained models on your local machine

