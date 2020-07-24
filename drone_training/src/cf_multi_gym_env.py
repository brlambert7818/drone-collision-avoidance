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
from crazyflie_driver.msg import Hover, GenericLogData, Position, FullState
import signal 
import os
import subprocess
import crazyflie


reg = register(
    id='CrazyflieMulti-v0',
    entry_point='cf_multi_gym_env:MultiCrazyflieEnv',
    )

class MultiCrazyflieEnv(gym.Env):

    def __init__(self, cf_id, gazebo, gazebo_process, cf_process):
        super(MultiCrazyflieEnv, self).__init__()

        self.cf_id = cf_id
        self.gazebo = gazebo
        self.gazebo_process = gazebo_process
        self.cf_process = cf_process

        self.rate = rospy.Rate(10)
        self.position_pub = rospy.Publisher('/cf' + str(self.cf_id) + '/cmd_full_state', FullState, queue_size=1)
        self.hover_pub = rospy.Publisher('/cf' + str(self.cf_id) + '/cmd_hover', Hover, queue_size=1)

        self.goal_position = np.array((2.5, 2.5, 5))

        # Gym spaces
        self.action_space = spaces.Box(low=np.array([-0.4, -0.4, 0.25]), high=np.array([0.4, 0.4, 9.5]), dtype=np.float32)
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


    def reset(self):
        """ Returns the drone to a starting postion to begin a new training episode. 

        Returns:
            ndarray (dtype=float, ndim=1): Array containing the current state observation.  
        """

        self.steps = 0

        # Connect to Crazyflie and enable high-level control
        self.cf = crazyflie.Crazyflie("cf" + str(self.cf_id), "/cf" + str(self.cf_id)) 
        self.cf.setParam("commander/enHighLevel", 1)

        reset_positions = self.random_position(-4, 5, 1, 10, 1)

        print('Start Reset')
        self.cf.takeoff(targetHeight = reset_positions[0][2], duration = 4)
        # time.sleep(4)

        # check if need to get individual values from np array
        self.cf.goTo(goal=[reset_positions[0][0], reset_positions[0][1], reset_positions[0][2]], yaw=0.0, duration=4)
        # time.sleep(4)
        print('End Reset')

        observation = self.get_observation()

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
                pose = rospy.wait_for_message('/cf' + str(self.cf_id) + '/local_position', GenericLogData, timeout=5)
            except:
                rospy.loginfo("Crazyflie pose not ready yet, retrying for getting robot pose")

        # get x,y,z position
        position = np.array(pose.values[:3])

        # get roll, pitch, and yaw Euler angles
        roll_pitch_yaw = np.array(pose.values[3:])

        # get angular velocities and linear accelerations
        imu = None
        while imu is None:
            try:
                imu = rospy.wait_for_message('/cf' + str(self.cf_id) + '/imu', Imu, timeout=5)
            except:
                rospy.loginfo("Crazyflie imu not ready yet, retrying for getting robot imu")

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
        self.hover_pub.publish(action_msg)
        time.sleep(0.3)

        observation = self.get_observation()
        is_flipped = False

        if abs(observation[-3]) >= 60:
            is_flipped = True
            print('FLIPPED....')

        # Analyze results 
        reward, is_terminal = self.reward(observation, is_flipped) 

        # restart simulation if drone has flipped
        if is_flipped:
            self.kill_sim()
            time.sleep(20)
            # self.gazebo_process, self.cf_process = self.launch_sim()
        
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

        cf_position = self.get_position()
        dist_to_goal = self.distance_between_points(cf_position, self.goal_position)
        reward = 0
        is_terminal = False

        # Reached goal
        if dist_to_goal < 1:
            reward += 50 
            is_terminal = True 
            print('REACHED GOAL.....')
        else:
            if self.steps == 128:
                is_terminal = True
            # Penalize based on distance to goal
            reward -= dist_to_goal / 500
            if is_flipped:
                # Penalize if drone has flipped over
                # reward -= 200 
                is_terminal = True
        
        return reward, is_terminal


    def close(self):
        self.kill_sim()


################################################################################
#                           Helper Functions
################################################################################


    def get_position(self):
        pose = None
        while pose is None:
            try:
                pose = rospy.wait_for_message('/cf' + str(self.cf_id) + '/local_position', GenericLogData, timeout=5)
            except:
                rospy.loginfo("Crazyflie pose not ready yet, retrying for getting robot pose")

        # get x,y,z position
        position = np.array(pose.values[:3])
        return position


    def process_action(self, action):
        """ Converts an array of actions into the necessary ROS msg type.

        Args:
            action (ndarray): Array containing the desired velocties along the 
            x, y, and z axes. 

        Returns:
            Hover: ROS msg type necessary to publish a velocity command.  
        """

        cf_posititon = self.get_position()
        
        # Bounce drone off right wall
        if cf_posititon[0] >= 4.5 and action[0] > 0:
            action[0] = 0
            cf_posititon = self.get_position()
        # Bounce drone off left wall
        elif cf_posititon[0] <= -4.5 and action[0] < 0:
            action[0] = 0
            cf_posititon = self.get_position()
        # Bounce drone off back wall
        if cf_posititon[1] >= 4.5 and action[1] > 0:
            action[1] = 0
            cf_posititon = self.get_position()
        # Bounce drone off front wall
        elif cf_posititon[1] <= -4.5 and action[1] < 0:
            action[1] = 0
            # cf_posititon = self.get_position()

        # Option 1: Hovering movements 
        action_msg = Hover()
        action_msg.vx = action[0] 
        action_msg.vy = action[1] 
        action_msg.zDistance = action[2]
        action_msg.yawrate = 0 

        return action_msg 


    def kill_sim(self):
        """ Terminates the Gazeo and Crazyflie processes.
        """
        rospy.loginfo('KILL SIM')
        os.killpg(self.cf_process, signal.SIGTERM)
        os.killpg(self.gazebo_process, signal.SIGTERM)


    def random_position(self, xy_min, xy_max, z_min, z_max, n_positions):
        """ Returns randomized x,y,x posititons for each drone. The function will 
        retry if the first generated position (the position for the primary agent) 
        is the same as the goal position.

        Args:
            xy_min (int): Inclusive minimum value for the x and y axes.
            xy_max (int): Exclusive maximum value for the x and y axes.
            z_min (int): Inclusive minimum value for the z axis.
            z_max (int): Exclusive maximum value for the z axis.
            n_positions (int): Number of positions to generate. 

        Returns:
            ndarray (dtype=float, ndim=(n_positions, 3)): [description]
        """
        while True:
            try:
                xy = np.random.randint(xy_min, xy_max, size=(n_positions, 2))
                z = np.random.randint(z_min, z_max, size=(n_positions, 1))
                xyz = np.append(xy, z, axis=1)
                if not np.array_equal(xyz[0], self.goal_position): return xyz
            except:
                pass


    def distance_between_points(self, point_a, point_b):
        """ Returns the Euclidean distance between two points.

        Args:
            point_a (list): (x, y, z) coordinates of the first point. 
            point_a (list): (x, y, z) coordinates of the second point. 

        Returns:
            float: Euclidean distance between the two points.
        """
        return np.linalg.norm(point_a - point_b)
