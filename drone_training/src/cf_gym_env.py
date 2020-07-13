#!/usr/bin/env python3

import gym
import rospy
import time
import numpy as np
import time
from gym import utils, spaces
from geometry_msgs.msg import Twist, Vector3Stamped, Pose
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty as EmptyTopicMsg
from gym.utils import seeding
from gym.envs.registration import register
from gazebo_connection import GazeboConnection
from crazyflie_driver.msg import Hover, GenericLogData, Position, FullState
import signal 
import os
import subprocess
import crazyflie

reg = register(
    id='Crazyflie-v0',
    entry_point='cf_gym_env:CrazyflieEnv',
    )

class CrazyflieEnv(gym.Env):

    def __init__(self):
        super(CrazyflieEnv, self).__init__()
        self.cf = None

        # Low-level Crazyflie control methods
        self.position_pub = rospy.Publisher('/cf1/cmd_full_state', FullState, queue_size=1)
        self.hover_pub = rospy.Publisher('/cf1/cmd_hover', Hover, queue_size=1)
        self.vel_pub = rospy.Publisher('/cf1/cmd_vel', Twist, queue_size=1)

        self.goal_position = np.array((5, 5, 5))

        # establishes connection with simulator
        self.gazebo_process, self.cf_process = self.launch_sim()

        # Gym spaces
        self.action_space = spaces.Box(low=np.array([-0.4, -0.4, 0.25]), high=np.array([0.4, 0.4, 19.5]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(12,), dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)
        self.steps = 0


    def seed(self, seed=None):
        """ Generates a random seed for the training environment.

        Args:
            seed (int, optional): Random seed number. Defaults to None.

        Returns:
            int: Random seed number. 
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def launch_sim(self):

        """ Executes bash commands to launch the Gazebo simulation, spawn a Crazyflie 
        UAV, and create a controller for the Crazyflie.

        Returns:
            bash process: Process corresponding to the Gazebo simulation
            bash process: Process corresponding to the Crazyflie model and controller 
        """
        rospy.loginfo('LAUNCH SIM')
        launch_gazebo_cmd = 'roslaunch crazyflie_gazebo crazyflie_sim.launch'
        gazebo_process = subprocess.Popen(launch_gazebo_cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        time.sleep(5)

        self.gazebo = GazeboConnection()

        cf_gazebo_path = '/home/brian/catkin_ws/src/sim_cf/crazyflie_gazebo/scripts'
        launch_controller_cmd = './run_cfs.sh'
        cf_process = subprocess.Popen(launch_controller_cmd, stdout=subprocess.PIPE, cwd=cf_gazebo_path, shell=True, preexec_fn=os.setsid)
        self.gazebo.unpauseSim()
        time.sleep(5)

        return gazebo_process, cf_process


    def kill_sim(self):
        """ Terminates the Gazeo and Crazyflie processes.
        """
        rospy.loginfo('KILL SIM')
        os.killpg(os.getpgid(self.cf_process.pid), signal.SIGTERM)
        os.killpg(os.getpgid(self.gazebo_process.pid), signal.SIGTERM)
        # subprocess.Popen('killall -9 gzserver gzclient'.split(' '), shell=False)
 

    def reset(self):
        """ Returns the drone to a starting postion to begin a new training episode. 

        Returns:
            ndarray (dtype=float, ndim=1): Array containing the current state observation.  
        """

        self.steps = 0

        # Connect to Crazyflie and enable high-level control
        self.cf = crazyflie.Crazyflie("cf1", "/cf1") 
        self.cf.setParam("commander/enHighLevel", 1)

        self.gazebo.unpauseSim()

        self.cf.takeoff(targetHeight = 2.0, duration = 0.3)
        time.sleep(4)

        observation = self.get_observation()
        self.gazebo.pauseSim()

        return observation


    def get_observation(self):
        """ Returns the current drone state consisting of the following: (x, y, z)
        positions, (x, y, z) angular velocities, (x, y, z) linear accelerations, 
        and (roll, pitch, yaw) Euler angles in degrees. 

        Returns:
            ndarray (dtype=float, ndim=1): Array containing the current state observation.  
        """

        pose = None
        while pose is None:
            try:
                pose = rospy.wait_for_message('/cf1/local_position', GenericLogData, timeout=5)
            except:
                rospy.loginfo("Crazyflie 1 pose not ready yet, retrying for getting robot pose")

        # get x,y,z position
        position = np.array(pose.values[:3])

        # get roll, pitch, and yaw Euler angles
        roll_pitch_yaw = np.array(pose.values[3:])

        # get angular velocities and linear accelerations
        imu = None
        while imu is None:
            try:
                imu = rospy.wait_for_message('/cf1/imu', Imu, timeout=5)
            except:
                rospy.loginfo("Crazyflie 1 imu not ready yet, retrying for getting robot imu")

        angular_velocity = np.array((imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z))
        linear_acceleration = np.array((imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z))

        return np.concatenate((position, angular_velocity, linear_acceleration, roll_pitch_yaw))


    def step(self, action):
        """ Executes an action and returns the resulting reward, state, and if 
        the episode has terminated.

        Args:
            action (Hover): Desired velocties along the x, y, and z axes. 

        Returns:
            ndarray (dtype=float, ndim=1): Array containing the current state observation.  
            int: Reward recieved from taking the action.
            bool: Whether or not the drone reached a terminal state as a result of
            of the action taken.
        """
        self.steps += 1

        action_msg = self.process_action(action)
        self.gazebo.unpauseSim()

        self.hover_pub.publish(action_msg)
        time.sleep(0.3)

        observation = self.get_observation()
        is_flipped = False

        # Check if drone has flipped (roll >= 60 degrees) 
        if observation[2] < 0.1 and abs(observation[-3]) >= 60:
            is_flipped = True
            print('FLIPPED....')

        # Used in box.world to avoid colliding with the walls
        # else:
        #     # Bounce drone off right wall
        #     if observation[0] > 9.5:
        #         bump_msg = FullState()
        #         bump_msg.pose.position.x = 9.0 
        #         bump_msg.pose.position.y = observation[1]
        #         bump_msg.pose.position.z = observation[2] 

        #         rate = rospy.Rate(10)
        #         i = 0
        #         while i < 5:
        #             self.position_pub.publish(bump_msg)
        #             i += 1
        #             rate.sleep()

        #         observation = self.get_observation()
        #     # Bounce drone off left wall
        #     if observation[0] < -9.5:
        #         bump_msg = FullState()
        #         bump_msg.pose.position.x = -9.0 
        #         bump_msg.pose.position.y = observation[1]
        #         bump_msg.pose.position.z = observation[2] 

        #         rate = rospy.Rate(10)
        #         i = 0
        #         while i < 5:
        #             self.position_pub.publish(bump_msg)
        #             i += 1
        #             rate.sleep()

        #         observation = self.get_observation()
        #     # Bounce drone off back wall
        #     if observation[1] > 9.5:
        #         bump_msg = FullState()
        #         bump_msg.pose.position.x = observation[0] 
        #         bump_msg.pose.position.y = 9.0
        #         bump_msg.pose.position.z = observation[2] 

        #         rate = rospy.Rate(10)
        #         i = 0
        #         while i < 5:
        #             self.position_pub.publish(bump_msg)
        #             i += 1
        #             rate.sleep()

        #         observation = self.get_observation()
        #     # Bounce drone off front wall
        #     if observation[1] < -9.5:
        #         bump_msg = FullState()
        #         bump_msg.pose.position.x = observation[0] 
        #         bump_msg.pose.position.y = -9.0 
        #         bump_msg.pose.position.z = observation[2] 

        #         rate = rospy.Rate(10)
        #         i = 0
        #         while i < 5:
        #             self.position_pub.publish(bump_msg)
        #             i += 1
        #             rate.sleep()

        #         observation = self.get_observation()

        # Analyze results 
        self.gazebo.pauseSim()
        reward, is_terminal = self.reward(observation, is_flipped) 

        # Restart simulation if drone has flipped
        if is_flipped:
            self.kill_sim()
            time.sleep(20)
            self.gazebo_process, self.cf_process = self.launch_sim()
        
        return observation, reward, is_terminal, {}
    

    def reward(self, observation, is_flipped):
        """ Returns the reward the drone will receive as a result of taking an action.

        Args:
            observation (ndarray, dtype=float, ndim=1): Array containing the current state observation.  
            is_flipped (bool): Whether or not the drone has flipped onto its back
            or side as a result of the previous action. 

        Returns:
            float: Reward for the drone to receive. 
            bool: Whether or not the drone has reached a terminal state as the 
            result of the previous action.
        """
        dist_to_goal = self.distance_between_points(observation[:3], self.goal_position)
        reward = 0
        is_terminal = False

        # Reward upward movement 
        # reward += 20 / abs(self.goal_position[2] - observation[2])
        # reward += observation[2]

        # Reached goal
        if dist_to_goal < 1:
            reward += 50 
            is_terminal = True 
            print('REACHED GOAL.....')
        else:
            if self.steps == 256:
                is_terminal = True
            # Penalize based on distance to goal
            reward -= dist_to_goal
            if is_flipped:
                # Penalize if drone has flipped over
                reward -= 50 
                is_terminal = True
        
        return reward, is_terminal

    def close(self):
        self.kill_sim()

    def process_action(self, action):
        """ Converts an array of actions into the necessary ROS msg type.

        Args:
            action (ndarray): Array containing the desired velocties along the 
            x, y, and z axes. 

        Returns:
            Hover: ROS msg type necessary to publish a velocity command.  
        """
        
        observation = self.get_observation()
        
        # Bounce drone off right wall
        if observation[0] > 9.5 and action[0] > 0:
            action[0] = 0
            observation = self.get_observation()
        # Bounce drone off left wall
        elif observation[0] < -9.5 and action[0] < 0:
            action[0] = 0
            observation = self.get_observation()

        # Bounce drone off back wall
        if observation[1] > 9.5 and action[1] > 0:
            action[1] = 0
            observation = self.get_observation()
        # Bounce drone off front wall
        elif observation[1] < -9.5 and action[1] < 0:
            action[1] = 0
            observation = self.get_observation()

        # Option 1: Hovering movements 
        action_msg = Hover()
        action_msg.vx = action[0] 
        action_msg.vy = action[1] 
        action_msg.zDistance = action[2]
        action_msg.yawrate = 0 

        # Option 2: Velocity movements 
        # action_msg = Twist()
        # action_msg.linear.x = action[0]
        # action_msg.linear.y = action[1]
        # action_msg.linear.z = action[2]
        # action_msg.angular.z = 0 

        # Option 3: Positon movements 
        # action_msg = FullState()
        # action_msg.pose.position.x = action[0] 
        # action_msg.pose.position.y = action[1]
        # action_msg.pose.position.z = action[2] 

        return action_msg 


    def distance_between_points(self, point_a, point_b):
        """ Returns the Euclidean distance between two points.

        Args:
            point_a (list): (x, y, z) coordinates of the first point. 
            point_a (list): (x, y, z) coordinates of the second point. 

        Returns:
            float: Euclidean distance between the two points.
        """
        return np.linalg.norm(point_a - point_b)


    def reset_position(self):
        """ Returns the drone to the starting position.
        """

        try:
            reset_msg = FullState()
            reset_msg.pose.position.x = 0
            reset_msg.pose.position.y = 0
            reset_msg.pose.position.z = 2

            rospy.loginfo("Go Home Start")
            rate = rospy.Rate(10)
            dist = np.inf
            sim_reset = False
            while dist > 0.1:
                self.position_pub.publish(reset_msg)
                cf_position = self.get_observation()[:3]
                dist = self.distance_between_points(cf_position, np.array((0, 0, 2)))

                # Check if drone has flipped over during reset
                # Ensure that the sim was not just reset or will enter reset loop
                # because the drone will spawn at z < -0.04 
                # if not sim_reset and cf_position[2] < -0.04:
                #     sim_reset = True
                #     self.kill_sim()
                #     time.sleep(20)
                #     self.gazebo_process, self.cf_process = self.launch_sim()

                rate.sleep()

            rospy.loginfo("Go Home completed")

        except rospy.ServiceException:
            print("Go Home not working")


    def step_high_level(self, action):
        """ Executes an action and returns the resulting reward, state, and if 
        the episode has terminated.

        Args:
            action (Hover): Desired velocties along the x, y, and z axes. 

        Returns:
            ndarray (dtype=float, ndim=1): Array containing the current state observation.  
            int: Reward recieved from taking the action.
            bool: Whether or not the drone reached a terminal state as a result of
            of the action taken.
        """

        # Stops drone from crashing into back wall 
        if action[0] > 9.5:
            action[0] = 9.5
        # Stops drone from crashing into front wall 
        elif action[0] < -9.5:
            action[0] = -9.5
        # Stops drone from crashing into right wall 
        elif action[1] > 9.5:
            action[1] = 9.5
        # Stops drone from crashing into left wall 
        elif action[1] < -9.5:
            action[1] = -9.5
        # Stops drone from crashing into floor
        elif action[2] < 0.5:
            action[2] = 0.5
        # Stops drone from crashing into ceiling 
        elif action[2] > 9.5:
            action[2] = 9.5

        self.gazebo.unpauseSim()

        self.cf.goTo(goal=[action[0], action[1], action[2]], yaw=0.0, duration=1)
        time.sleep(0.3)

        observation = self.get_observation()
        is_flipped = False

        # check if drone has flipped during step (i.e. roll >= 60 degrees) 
        if observation[2] < 0.1 and abs(observation[-3]) >= 60:
            is_flipped = True
            print('FLIPPED....')

        # analyze results 
        self.gazebo.pauseSim()
        reward, is_terminal = self.reward(observation, is_flipped) 

        # restart simulation if drone has flipped
        if is_flipped:
            self.kill_sim()
            time.sleep(20)
            self.gazebo_process, self.cf_process = self.launch_sim()
        
        return observation, reward, is_terminal, {}