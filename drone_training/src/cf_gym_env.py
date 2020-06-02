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

reg = register(
    id='Crazyflie-v0',
    entry_point='cf_gym_env:CrazyflieEnv',
    # max_episode_steps=50,
    )

class CrazyflieEnv(gym.Env):

    def __init__(self):

        # rate
        self.rate = rospy.Rate(10)
        # self.n = 3  # 3 drones

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

        # in common for every drone
        self.running_step = rospy.get_param("/drone1/running_step")
        self.max_incl = rospy.get_param("/drone1/max_incl")
        self.max_altitude = rospy.get_param("/drone1/max_altitude")
        self.max_speed = rospy.get_param("/drone1/speed_value")

        # stablishes connection with simulator
        self.gazebo = GazeboConnection()

        # spaces
        self.action_space = spaces.Box(low=-self.max_speed, high=+self.max_speed, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(3,), dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)

        # first time
        self.first = True

        self.seed()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.gazebo.unpauseSim()

        if not self.first:
            self.gazebo.resetSim_definitive()
        self.first = False


        self.check_topic_publishers_connection()  # SIMULATION RUNNING
        self.takeoff_sequence()  # SIMULATION RUNNING
        print('takeoff finished')

        data_pose = self.get_observation()  # SIMULATION RUNNING
        # orientation = self.orientation(data_imu)
        observation = np.array(data_pose.values[:3])
        # obs = np.concatenate((position, orientation))

        self.gazebo.pauseSim()

        return observation


    # def orientation(self, ros_data):
    #     return np.array([ros_data.orientation.x,ros_data.orientation.y,ros_data.orientation.z, ros_data.orientation.w])


    def process_action(self, action):
        action = np.clip(action, a_min = -0.3, a_max= 0.3)
        vel_cmd = Hover()
        vel_cmd.vx = action[0]
        vel_cmd.vy = action[1]
        vel_cmd.yawrate = action[2]
        vel_cmd.zDistance = action[3]
        # vel_cmd.yawrate = 0 
        # vel_cmd.zDistance = 5 
        return vel_cmd


    def step(self, action):

        vel_cmd = self.process_action(action)

        self.gazebo.unpauseSim()

        self.vel_pub.publish(vel_cmd)
        time.sleep(self.running_step)

        data_pose = self.get_observation()

        # ANALYZE THE RESULTS
        self.gazebo.pauseSim()

        reward, is_terminal = self.reward(data_pose)
        # orientation = self.orientation(data_imu)
        observation = np.array(data_pose.values[:3])
        # obs = np.concatenate((position, orientation))

        return observation, reward, is_terminal, {}


    def get_observation(self):
        data_pose = None
        while data_pose is None:
            try:
                data_pose = rospy.wait_for_message('/cf1/local_position', GenericLogData, timeout=5)
            except:
                rospy.loginfo("Crazyflie 1 pose not ready yet, retrying for getting robot pose")

        # data_imu = None
        # while data_imu is None:
        #     try:
        #         data_imu = rospy.wait_for_message('/cf1/imu', Imu, timeout=5)
        #     except:
        #         rospy.loginfo("Crazyflie 1 imu not ready yet, retrying for getting robot imu")

        return data_pose


    def distance_between_points(self, p_init, p_end):
        a = np.array((p_init.x, p_init.y, p_init.z))
        b = np.array((p_end.x, p_end.y, p_end.z))
        dist = np.linalg.norm(a - b)
        return dist


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


    def into_pose(self, data_position):
        current_pose = Pose()
        current_pose.position.x = data_position.values[0]
        current_pose.position.y = data_position.values[1]
        current_pose.position.z = data_position.values[2]
        return current_pose


    def reward(self, position):

        current_pose = self.into_pose(position)
        dist_to_goal = self.distance_between_points(current_pose.position, self.goal.position)

        if dist_to_goal < 1:
            return 50, True
        else:
            return -dist_to_goal, False
