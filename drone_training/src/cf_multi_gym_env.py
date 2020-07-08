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


reg = register(
    id='Crazyflie-v1',
    entry_point='cf_multi_gym_env:MultiCrazyflieEnv',
    )

class MultiCrazyflieEnv(gym.Env):

    def __init__(self, cf_id, gazebo, gazebo_process, cf_process):

        self.cf_id = cf_id
        self.gazebo = gazebo
        self.gazebo_process = gazebo_process
        self.cf_process = cf_process

        self.rate = rospy.Rate(10)
        self.position_pub = rospy.Publisher('/cf' + str(self.cf_id) + '/cmd_full_state', FullState, queue_size=1)
        self.vel_pub = rospy.Publisher('/cf' + str(self.cf_id) + '/cmd_hover', Hover, queue_size=1)

        # gets training parameters from param server
        self.speed_value = rospy.get_param("drone1/speed_value")
        self.goal = Pose()
        self.goal.position.x = rospy.get_param("/goal/desired_pose/x")
        self.goal.position.y = rospy.get_param("/goal/desired_pose/y")
        self.goal.position.z = rospy.get_param("/goal/desired_pose/z")
        self.goal_position = np.array((self.goal.position.x, self.goal.position.y, self.goal.position.z))
        self.running_step = rospy.get_param("/drone1/running_step")
        self.max_incl = rospy.get_param("/drone1/max_incl")
        self.max_altitude = rospy.get_param("/drone1/max_altitude")
        self.max_speed = rospy.get_param("/drone1/speed_value")

        # Get reset position
        self.reset_x = rospy.get_param('reset_positions/cf' + str(self.cf_id) + '/x') 
        self.reset_y = rospy.get_param('reset_positions/cf' + str(self.cf_id) + '/y') 
        self.reset_z = rospy.get_param('reset_positions/cf' + str(self.cf_id) + '/z') 

        # spaces
        self.action_space = spaces.Box(low=-self.max_speed, high=+self.max_speed, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(12,), dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)

        self.is_launch = True
        self.seed()


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

        self.gazebo.unpauseSim()
        self.sim_reset = True

        if not self.is_launch:
            # reset yaw to avoid flipping during reset
            final_position = self.get_observation()[:3]
            stabilize_msg = FullState()
            stabilize_msg.pose.position.x = final_position[0]
            stabilize_msg.pose.position.y = final_position[1]
            stabilize_msg.pose.position.z = final_position[2]

            rate = rospy.Rate(10)
            i = 0
            while i < 20:
                self.position_pub.publish(stabilize_msg)
                i += 1
                rate.sleep()

        # reset to starting positon after stabilization
        self.reset_position()
        observation = self.get_observation()
        self.gazebo.pauseSim()
        self.is_launch = False 

        return observation

    
    def reset_position(self):
        """ Returns the drone to the starting position.
        """
        try:
            reset_msg = FullState()
            reset_msg.pose.position.x = self.reset_x 
            reset_msg.pose.position.y = self.reset_y 
            reset_msg.pose.position.z = self.reset_z 

            rospy.loginfo("Go Home Start")
            rate = rospy.Rate(10)
            dist = np.inf
            sim_reset = False
            while dist > 0.1:
                self.position_pub.publish(reset_msg)
                cf_position = self.get_observation()[:3]
                dist = self.distance_between_points(cf_position, np.array((self.reset_x, self.reset_y, self.reset_z)))

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


    def kill_sim(self):
        """ Terminates the Gazeo and Crazyflie processes.
        """
        rospy.loginfo('KILL SIM')
        os.killpg(self.cf_process, signal.SIGTERM)
        os.killpg(self.gazebo_process, signal.SIGTERM)

        self.is_launch = True


    def process_action(self, action):
        """ Converts an array of actions into the necessary ROS msg type.

        Args:
            action (ndarray): Array containing the desired velocties along the 
            x, y, and z axes. 

        Returns:
            Hover: ROS msg type necessary to publish a velocity command.  
        """

        vel_cmd = Hover()
        vel_cmd.vx = action[0] 
        vel_cmd.vy = action[1] 
        vel_cmd.zDistance = action[2]
        vel_cmd.yawrate = 0 
        return vel_cmd


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

        self.sim_reset = False
        vel_cmd = self.process_action(action)

        # self.gazebo.unpauseSim()

        self.vel_pub.publish(vel_cmd)
        time.sleep(self.running_step)

        observation = self.get_observation()
        is_flipped = False

        # check if drone has flipped during step (i.e. roll >= 60 degrees) 
        if abs(observation[-3]) >= 60:
            print('FLIPPED....')
            is_flipped = True
            print('state: ', observation)

        else:
            if observation[2] < 0.7:
                # stops cf from getting legs stuck in ground plane 
                bump_msg = FullState()
                bump_msg.pose.position.x = observation[0]
                bump_msg.pose.position.y = observation[1]
                bump_msg.pose.position.z = 0.7 

                rate = rospy.Rate(10)
                i = 0
                while i < 5:
                    self.position_pub.publish(bump_msg)
                    i += 1
                    rate.sleep()

                # new position due to 'barrier' in env
                observation = self.get_observation()


            # bounce drone off right wall
            if observation[0] > 9.5:
                bump_msg = FullState()
                bump_msg.pose.position.x = 9.0 
                bump_msg.pose.position.y = observation[1]
                bump_msg.pose.position.z = observation[2] 

                rate = rospy.Rate(10)
                i = 0
                while i < 5:
                    self.position_pub.publish(bump_msg)
                    i += 1
                    rate.sleep()

                # new position due to 'barrier' in env
                observation = self.get_observation()
            # bounce drone off left wall
            if observation[0] < -9.5:
                bump_msg = FullState()
                bump_msg.pose.position.x = -9.0 
                bump_msg.pose.position.y = observation[1]
                bump_msg.pose.position.z = observation[2] 

                rate = rospy.Rate(10)
                i = 0
                while i < 5:
                    self.position_pub.publish(bump_msg)
                    i += 1
                    rate.sleep()

                # new position due to 'barrier' in env
                observation = self.get_observation()
            # bounce drone off back wall
            if observation[1] > 9.5:
                bump_msg = FullState()
                bump_msg.pose.position.x = observation[0] 
                bump_msg.pose.position.y = 9.0
                bump_msg.pose.position.z = observation[2] 

                rate = rospy.Rate(10)
                i = 0
                while i < 5:
                    self.position_pub.publish(bump_msg)
                    i += 1
                    rate.sleep()

                # new position due to 'barrier' in env
                observation = self.get_observation()
            # bounce drone off front wall
            if observation[1] < -9.5:
                bump_msg = FullState()
                bump_msg.pose.position.x = observation[0] 
                bump_msg.pose.position.y = -9.0 
                bump_msg.pose.position.z = observation[2] 

                rate = rospy.Rate(10)
                i = 0
                while i < 5:
                    self.position_pub.publish(bump_msg)
                    i += 1
                    rate.sleep()

                # new position due to 'barrier' in env
                observation = self.get_observation()


        # analyze results 
        # self.gazebo.pauseSim()
        reward, is_terminal = self.reward(observation, is_flipped) 

        # restart simulation if drone has flipped
        if is_flipped:
            self.kill_sim()
            time.sleep(20)
            # self.gazebo_process, self.cf_process = self.launch_sim()
        
        return observation, reward, is_terminal, {}


    def get_observation(self):
        """ Returns the current drone state consisting of the following: (x, y, z)
        positions, (x, y, z) angular velocities, (x, y, z) linear accelerations, 
        and (roll, pitch, yaw) Euler angles in degrees. 

        Returns:
            ndarray (dtype=float, ndim=1): Array containing the current state observation.  
        """

        pose = None
        t = None
        while pose is None:
            try:
                pose = rospy.wait_for_message('/cf' + str(self.cf_id) + '/local_position', GenericLogData, timeout=5)
            except:
                t = None 
                # rospy.loginfo("Crazyflie 1 pose not ready yet, retrying for getting robot pose")

        # get x,y,z position
        position = np.array(pose.values[:3])

        # get roll, pitch, and yaw Euler angles
        roll_pitch_yaw = np.array(pose.values[3:])

        # get angular velocities and linear accelerations
        imu = None
        f = None
        while imu is None:
            try:
                imu = rospy.wait_for_message('/cf' + str(self.cf_id) + '/imu', Imu, timeout=5)
            except:
                f = None
                # rospy.loginfo("Crazyflie 1 imu not ready yet, retrying for getting robot imu")

        angular_velocity = np.array((imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z))
        linear_acceleration = np.array((imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z))

        return np.concatenate((position, angular_velocity, linear_acceleration, roll_pitch_yaw))


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

        # reward altitude
        # reward += 20 / abs(self.goal_position[2] - observation[2])
        reward += 5*observation[2]

        if dist_to_goal < 1:
            reward += 400 
            is_terminal = True 
            print('REACHED GOAL.....')
        else:
            reward -= dist_to_goal
            if is_flipped:
                reward -= 50 
                is_terminal = True
        
        return reward, is_terminal


    def distance_between_points(self, point_a, point_b):
        """ Returns the Euclidean distance between two points.

        Args:
            point_a (list): (x, y, z) coordinates of the first point. 
            point_a (list): (x, y, z) coordinates of the second point. 

        Returns:
            float: Euclidean distance between the two points.
        """
        return np.linalg.norm(point_a - point_b)


    def check_topic_publishers_connection(self):
        """ Ensures that connection has been established to the ROS publishers
        used for movement control. 
        """
        rate = rospy.Rate(10)
        while (self.position_pub.get_num_connections() == 0):
            rate.sleep()
        while (self.vel_pub.get_num_connections() == 0):
            rate.sleep()
