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

        # postion where the drone begins each episode
        self.init_position = np.array((0, 0, 5))

        # gets training parameters from param server
        self.speed_value = rospy.get_param("drone1/speed_value")
        self.goal = Pose()
        self.goal.position.x = rospy.get_param("/goal/desired_pose/x")
        self.goal.position.y = rospy.get_param("/goal/desired_pose/y")
        self.goal.position.z = rospy.get_param("/goal/desired_pose/z")
        self.running_step = rospy.get_param("/drone1/running_step")
        self.max_incl = rospy.get_param("/drone1/max_incl")
        self.max_altitude = rospy.get_param("/drone1/max_altitude")
        self.max_speed = rospy.get_param("/drone1/speed_value")

        # establishes connection with simulator
        # self.gazebo_process = self.launch_gazebo() 
        # self.gazebo = GazeboConnection()
        # self.controller_process = self.launch_controller()
        self.gazebo_process, self.cf_process = self.launch_sim()

        # spaces
        self.action_space = spaces.Box(low=-self.max_speed, high=+self.max_speed, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(3,), dtype=np.float32)
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
            final_position = np.array(self.get_observation().values[:3])

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

            # TO DO: catch for obs[3] flip during reset

        # reset to starting positon after stabilization
        self.gazebo.reset_position()

        data_pose = self.get_observation()  # SIMULATION RUNNING
        observation = np.array(data_pose.values[:3])

        self.gazebo.pauseSim()

        self.is_launch = False 

        return observation


    def launch_gazebo(self):
        launch_gazebo_cmd = 'roslaunch crazyflie_gazebo crazyflie_sim.launch'
        gazebo_process = subprocess.Popen(launch_gazebo_cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        time.sleep(5)
        return gazebo_process


    def launch_controller(self):
        cf_gazebo_path = '/home/brian/catkin_ws/src/sim_cf/crazyflie_gazebo/scripts'
        launch_controller_cmd = './run_cfs.sh'
        cf_process = subprocess.Popen(launch_controller_cmd, stdout=subprocess.PIPE, cwd=cf_gazebo_path, shell=True, preexec_fn=os.setsid)
        self.gazebo.unpauseSim()
        time.sleep(5)
        return cf_process


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
        # action_clip = np.clip(action, a_min = -0.3, a_max= 0.3)
        action_clip = action
        vel_cmd = Hover()
        vel_cmd.vx = action_clip[0]
        vel_cmd.vy = action_clip[1]
        vel_cmd.yawrate = action_clip[2]
        vel_cmd.zDistance = action[3]
        return vel_cmd


    def step(self, action):

        vel_cmd = self.process_action(action)

        self.gazebo.unpauseSim()

        self.vel_pub.publish(vel_cmd)
        time.sleep(self.running_step)

        pose = self.get_observation()
        observation = np.array(pose.values[:3])
        is_flipped = False

        # stops cf from getting legs stuck in ground plane 
        if observation[2] < 0.8:
            # restart simulation if drone has flipped
            if observation[2] < -0.04:
                is_flipped = True
                self.kill_sim()
                time.sleep(20)
                self.gazebo_process, self.cf_process = self.launch_sim()
            else:
                bump_msg = FullState()
                bump_msg.pose.position.x = observation[0]
                bump_msg.pose.position.y = observation[1]
                bump_msg.pose.position.z = 0.8 

                rate = rospy.Rate(10)
                i = 0
                while i < 4:
                    self.takeoff_pub.publish(bump_msg)
                    i += 1
                    rate.sleep()

            pose = self.get_observation()
            observation = np.array(pose.values[:3])

        # ANALYZE THE RESULTS
        self.gazebo.pauseSim()

        reward, is_terminal = self.reward(pose, is_flipped) 
        
        return observation, reward, is_terminal, {}


    def get_observation(self):
        pose = None
        while pose is None:
            try:
                pose = rospy.wait_for_message('/cf1/local_position', GenericLogData, timeout=5)
            except:
                rospy.loginfo("Crazyflie 1 pose not ready yet, retrying for getting robot pose")

        return pose


    def distance_between_points(self, point_a, point_b):
        a = np.array((point_a.x, point_a.y, point_a.z))
        b = np.array((point_b.x, point_b.y, point_b.z))
        distance = np.linalg.norm(a - b)
        return distance


    def check_topic_publishers_connection(self):
        rate = rospy.Rate(10)

        while (self.takeoff_pub.get_num_connections() == 0):
            rate.sleep()

        while (self.vel_pub.get_num_connections() == 0):
            rate.sleep()


    def reset_commands(self):
        vel_cmd = Hover()
        vel_cmd.vx = 0.0
        vel_cmd.vy = 0.0
        vel_cmd.zDistance = 0.0
        vel_cmd.yawrate = 0.0
        self.vel_pub.publish(vel_cmd)


    def takeoff_sequence(self):
        takeoff_msg = FullState()
        takeoff_msg.pose.position.z = 5.0 
        takeoff_msg.pose.position.y = 0
        takeoff_msg.pose.position.x = 0

        rate = rospy.Rate(10)
        i = 0
        while i < 50:
            self.takeoff_pub.publish(takeoff_msg)
            i += 1
            rate.sleep()


    def pose(self, data_position):
        current_pose = Pose()
        current_pose.position.x = data_position.values[0]
        current_pose.position.y = data_position.values[1]
        current_pose.position.z = data_position.values[2]
        return current_pose


    def reward(self, position, is_flipped):
        current_pose = self.pose(position)
        dist_to_goal = self.distance_between_points(current_pose.position, self.goal.position)

        reward = 0
        is_terminal = False

        if dist_to_goal < 1:
            reward += 50
            is_terminal = True 
        else:
            reward -= dist_to_goal
            if is_flipped:
                reward -= 10
                is_terminal = True
        
        return reward, is_terminal

