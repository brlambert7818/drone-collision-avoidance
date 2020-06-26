#!/usr/bin/env python

import gym
import rospy
import time
import numpy as np
import tf
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

reg = register(
    id='Crazyflie-v0',
    entry_point='cf_gym_env:CrazyflieEnv',
    # max_episode_steps=50,
    )

class CrazyflieEnv(gym.Env):

    def __init__(self):

        # rate
        self.rate = rospy.Rate(10)
        # takeoff
        self.takeoff_pub = rospy.Publisher('/cf1/cmd_full_state', FullState, queue_size=1)
        # execute actions
        self.vel_pub = rospy.Publisher('/cf1/cmd_hover', Hover, queue_size=1)

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

        # establishes connection with simulator
        self.gazebo_process, self.cf_process = self.launch_sim()

        # spaces
        self.action_space = spaces.Box(low=-self.max_speed, high=+self.max_speed, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(12,), dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)

        self.is_launch = True

        self.seed()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):

        self.gazebo.unpauseSim()

        if not self.is_launch:
            # reset yaw to avoid flipping during reset
            # final_position = np.array(self.get_observation().values[:3])
            final_position = self.get_observation()[:3]

            stabilize_msg = FullState()
            stabilize_msg.pose.position.x = final_position[0]
            stabilize_msg.pose.position.y = final_position[1]
            stabilize_msg.pose.position.z = final_position[2]

            rate = rospy.Rate(10)
            i = 0
            while i < 20:
                self.takeoff_pub.publish(stabilize_msg)
                i += 1
                rate.sleep()

        # reset to starting positon after stabilization
        self.reset_position()

        observation = self.get_observation()

        self.gazebo.pauseSim()

        self.is_launch = False 

        return observation

    
    def reset_position(self):
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
                self.takeoff_pub.publish(reset_msg)
                cf_position = self.get_observation()[:3]
                dist = self.distance_between_points(cf_position, np.array((0, 0, 2)))

                # Check if drone has flipped over during reset
                # Ensure that the sim was not just reset or will enter reset loop
                # because the drone will spawn at z < -0.04 
                if not sim_reset and cf_position[2] < -0.04:
                    sim_reset = True
                    self.kill_sim()
                    time.sleep(20)
                    self.gazebo_process, self.cf_process = self.launch_sim()

                rate.sleep()

            rospy.loginfo("Go Home completed")

        except rospy.ServiceException, e:
            print("Go Home not working")


    def launch_sim(self):
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
        rospy.loginfo('KILL SIM')
        os.killpg(os.getpgid(self.cf_process.pid), signal.SIGTERM)
        os.killpg(os.getpgid(self.gazebo_process.pid), signal.SIGTERM)
        # self.gazebo_process.kill()
        # self.cf_process.kill()

        self.is_launch = True


    def process_action(self, action):
        action_clip = np.clip(action, a_min = -0.3, a_max= 0.3)
        # action_clip = action 
        vel_cmd = Hover()
        # clip x and y actions to prevent flipping
        vel_cmd.vx = action_clip[0]
        vel_cmd.vy = action_clip[1]
        vel_cmd.zDistance = action[2]
        vel_cmd.yawrate = 0.0 
        return vel_cmd


    def step(self, action, training_done):

        vel_cmd = self.process_action(action)

        self.gazebo.unpauseSim()

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
            if observation[2] < 0.75:
                # check if drone is at rest
                if observation[2] < -0.04 and (abs(observation[3]) < 0.005 and abs(observation[4]) < 0.005):
                    print('STATIONARY....')
                    is_flipped = True
                # stops cf from getting legs stuck in ground plane 
                else:
                    bump_msg = FullState()
                    bump_msg.pose.position.x = observation[0]
                    bump_msg.pose.position.y = observation[1]
                    bump_msg.pose.position.z = 1.0 

                    rate = rospy.Rate(10)
                    i = 0
                    while i < 5:
                        self.takeoff_pub.publish(bump_msg)
                        i += 1
                        rate.sleep()

                    # new position due to 'barrier' in env
                    observation = self.get_observation()
            if observation[0] > 9.5:
                    bump_msg = FullState()
                    bump_msg.pose.position.x = 9.0 
                    bump_msg.pose.position.y = observation[1]
                    bump_msg.pose.position.z = observation[2] 

                    rate = rospy.Rate(10)
                    i = 0
                    while i < 5:
                        self.takeoff_pub.publish(bump_msg)
                        i += 1
                        rate.sleep()

                    # new position due to 'barrier' in env
                    observation = self.get_observation()
            if observation[0] < -9.5:
                    bump_msg = FullState()
                    bump_msg.pose.position.x = -9.0 
                    bump_msg.pose.position.y = observation[1]
                    bump_msg.pose.position.z = observation[2] 

                    rate = rospy.Rate(10)
                    i = 0
                    while i < 5:
                        self.takeoff_pub.publish(bump_msg)
                        i += 1
                        rate.sleep()

                    # new position due to 'barrier' in env
                    observation = self.get_observation()
            if observation[1] > 9.5:
                    bump_msg = FullState()
                    bump_msg.pose.position.x = observation[0] 
                    bump_msg.pose.position.y = 9.0
                    bump_msg.pose.position.z = observation[2] 

                    rate = rospy.Rate(10)
                    i = 0
                    while i < 5:
                        self.takeoff_pub.publish(bump_msg)
                        i += 1
                        rate.sleep()

                    # new position due to 'barrier' in env
                    observation = self.get_observation()
            if observation[1] < -9.5:
                    bump_msg = FullState()
                    bump_msg.pose.position.x = observation[0] 
                    bump_msg.pose.position.y = -9.0 
                    bump_msg.pose.position.z = observation[2] 

                    rate = rospy.Rate(10)
                    i = 0
                    while i < 5:
                        self.takeoff_pub.publish(bump_msg)
                        i += 1
                        rate.sleep()

                    # new position due to 'barrier' in env
                    observation = self.get_observation()


        # analyze results 
        self.gazebo.pauseSim()
        reward, is_terminal = self.reward(observation, is_flipped) 

        # restart simulation if drone has flipped
        if is_flipped:
            self.kill_sim()
            time.sleep(20)

            # Only restart the sim if there are training episodes remaining
            if not training_done:
                self.gazebo_process, self.cf_process = self.launch_sim()
        
        # KIll sim if training is done and drone hasn't crashed
        elif training_done:
            self.kill_sim()
        
        return observation, reward, is_terminal, {}


    def get_observation(self):
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
        # rospy.loginfo(roll_pitch_yaw)

        # get angular velocities and linear accelerations
        imu = None
        while imu is None:
            try:
                imu = rospy.wait_for_message('/cf1/imu', Imu, timeout=5)
            except:
                rospy.loginfo("Crazyflie 1 imu not ready yet, retrying for getting robot imu")

        angular_velocity = np.array((imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z))
        linear_acceleration = np.array((imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z))

        # get roll, pitch, and yaw Euler angles
        # (roll, pitch, yaw) = tf.transformations.euler_from_quaternion([imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w]) 

        return np.concatenate((position, angular_velocity, linear_acceleration, roll_pitch_yaw))


    def distance_between_points(self, point_a, point_b):
        return np.linalg.norm(point_a - point_b)


    def check_topic_publishers_connection(self):
        rate = rospy.Rate(10)
        while (self.takeoff_pub.get_num_connections() == 0):
            rate.sleep()
        while (self.vel_pub.get_num_connections() == 0):
            rate.sleep()


    def pose(self, data_position):
        current_pose = Pose()
        current_pose.position.x = data_position.values[0]
        current_pose.position.y = data_position.values[1]
        current_pose.position.z = data_position.values[2]
        return current_pose


    def reward(self, observation, is_flipped):
        dist_to_goal = self.distance_between_points(observation[:3], self.goal_position)
        reward = 0
        is_terminal = False

        # reward altitude
        # reward += 20 / abs(self.goal_position[2] - observation[2])
        reward += 10 * observation[2]

        if dist_to_goal < 1:
            reward += 50
            is_terminal = True 
            print('REACHED GOAL.....')
        else:
            reward -= dist_to_goal
            if is_flipped:
                reward -= 10
                is_terminal = True
        
        return reward, is_terminal

