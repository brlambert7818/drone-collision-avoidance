#!/usr/bin/env python

# test comment

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
    id='QuadcopterLiveShow-v0',
    entry_point='myquadcopter_env3deep:MultiQuadCopterEnvDeep',
    max_episode_steps=50,
    )

class MultiQuadCopterEnvDeep(gym.Env):

    def __init__(self):

        # rate
        self.rate = rospy.Rate(10)
        self.n = 3  # 3 drones

        # takeoff
        self.takeoff_pub1 = rospy.Publisher('/cf1/cmd_full_state', FullState, queue_size=1)
        self.takeoff_pub2 = rospy.Publisher('/cf2/cmd_full_state', FullState, queue_size=1)
        self.takeoff_pub3 = rospy.Publisher('/cf3/cmd_full_state', FullState, queue_size=1)

        self.vel_pub1 = rospy.Publisher('/cf1/cmd_hover', Hover, queue_size=1)
        self.vel_pub2 = rospy.Publisher('/cf2/cmd_hover', Hover, queue_size=1)
        self.vel_pub3 = rospy.Publisher('/cf3/cmd_hover', Hover, queue_size=1)

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

        # formation radius
        self.formation_radius = 1

        # stablishes connection with simulator
        self.gazebo = GazeboConnection()

        # spaces
        self.action_space = spaces.Box(low=-self.max_speed, high=+self.max_speed, shape=(2,), dtype=np.float32)
        self.action_space_n = [self.action_space, self.action_space, self.action_space]

        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(9,), dtype=np.float32)
        self.observation_space_n = [self.observation_space, self.observation_space, self.observation_space]

        self.reward_range = (-np.inf, np.inf)

        # first time
        self.first = True

        self._seed()


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):

        self.gazebo.unpauseSim()

        if not self.first:
            self.gazebo.resetSim_definitive4()
        self.first = False


        self.check_topic_publishers_connection()  # SIMULATION RUNNING
        self.takeoff_sequence()  # SIMULATION RUNNING

        data_pose1, data_imu1, data_pose2, data_imu2, data_pose3, data_imu3 = self.take_observation()  # SIMULATION RUNNING

        observation_n = []

        orientation1 = self.orientation(data_imu1)
        orientation2 = self.orientation(data_imu2)
        orientation3 = self.orientation(data_imu3)


        observation1 =np.concatenate( (np.array(data_pose1.values[:2]), np.subtract(data_pose2.values[:2], data_pose1.values[:2]),
                        np.subtract(data_pose3.values[:2], data_pose1.values[:1]))) # SIMULATION RUNNING
        observation2 = np.concatenate((np.array(data_pose2.values[:2]), np.subtract(data_pose1.values[:2], data_pose2.values[:2]),
                        np.subtract(data_pose3.values[:2], data_pose2.values[:1]))) # SIMULATION RUNNING
        # SIMULATION RUNNING
        observation3 = np.concatenate((np.array(data_pose3.values[:2]), np.subtract(data_pose1.values[:2], data_pose3.values[:2]),
                        np.subtract(data_pose2.values[:2], data_pose3.values[:1])) )# SIMULATION RUNNING
        # SIMULATION RUNNING

        observation1 = np.concatenate((observation1,orientation1))
        observation2 = np.concatenate((observation2, orientation2))
        observation3 = np.concatenate((observation3, orientation3))

        observation_n.append(observation1)
        observation_n.append(observation2)
        observation_n.append(observation3)

        self.gazebo.pauseSim()

        return observation_n

    def orientation(selfs, ros_data):
        return np.array([ros_data.orientation.x,ros_data.orientation.y,ros_data.orientation.z, ros_data.orientation.w])

    def process_action(self, action, i):

        action = np.clip(action, a_min = -0.3, a_max= 0.3)
        if i == 1:
            vel_cmd = Hover()
            vel_cmd.vx = action[0]
            vel_cmd.vy = action[1]
            vel_cmd.yawrate = 0.0
            vel_cmd.zDistance = 0.7
        elif i == 2:
            vel_cmd = Hover()
            vel_cmd.vx = action[0]
            vel_cmd.vy = action[1]
            vel_cmd.yawrate = 0.0
            vel_cmd.zDistance = 1
        else:
            vel_cmd = Hover()
            vel_cmd.vx = action[0]
            vel_cmd.vy = action[1]
            vel_cmd.yawrate = 0.0
            vel_cmd.zDistance = 1.3
        return vel_cmd

    def _step(self, action):

        vel_cmd1 = self.process_action(action[0],1)
        vel_cmd2 = self.process_action(action[1],2)
        vel_cmd3 = self.process_action(action[2],3)

        self.gazebo.unpauseSim()

        self.vel_pub1.publish(vel_cmd1)
        self.vel_pub2.publish(vel_cmd2)
        self.vel_pub3.publish(vel_cmd3)
        time.sleep(self.running_step)

        data_pose1, data_imu1, data_pose2, data_imu2, data_pose3, data_imu3 = self.take_observation()

        # ANALYZE THE RESULTS
        self.gazebo.pauseSim()

        # reward
        reward1, done1 = self.reward(data_pose1, data_imu1, data_pose2, data_pose3)
        reward2, done2 = self.reward(data_pose2, data_imu2, data_pose1, data_pose3)
        reward3, done3 = self.reward(data_pose3, data_imu3, data_pose1, data_pose2)

        # shared reward for cooperative
        reward_n = [reward1, reward2, reward3]
        reward_sum = np.sum(reward_n)
        reward_n = [reward_sum, reward_sum, reward_sum]

        orientation1 = self.orientation(data_imu1)
        orientation2 = self.orientation(data_imu2)
        orientation3 = self.orientation(data_imu3)

        observation1 = np.concatenate(
            (np.array(data_pose1.values[:2]), np.subtract(data_pose2.values[:2], data_pose1.values[:2]),
             np.subtract(data_pose3.values[:2], data_pose1.values[:1])))  # SIMULATION RUNNING
        observation2 = np.concatenate(
            (np.array(data_pose2.values[:2]), np.subtract(data_pose1.values[:2], data_pose2.values[:2]),
             np.subtract(data_pose3.values[:2], data_pose2.values[:1])))  # SIMULATION RUNNING
        # SIMULATION RUNNING
        observation3 = np.concatenate(
            (np.array(data_pose3.values[:2]), np.subtract(data_pose1.values[:2], data_pose3.values[:2]),
             np.subtract(data_pose2.values[:2], data_pose3.values[:1])))  # SIMULATION RUNNING
        # SIMULATION RUNNING

        observation1 = np.concatenate((observation1, orientation1))
        observation2 = np.concatenate((observation2, orientation2))
        observation3 = np.concatenate((observation3, orientation3))

        # SIMULATION RUNNING
        observation_n = []
        observation_n.append(observation1)
        observation_n.append(observation2)
        observation_n.append(observation3)

        done_n = [done1, done2, done3]

        return observation_n, reward_n, done_n, {}

    def take_observation(self):
        data_pose1 = None
        while data_pose1 is None:
            try:
                data_pose1 = rospy.wait_for_message('/cf1/local_position', GenericLogData, timeout=5)
            except:
                rospy.loginfo("Crazyflie 1 pose not ready yet, retrying for getting robot pose")

        data_imu1 = None
        while data_imu1 is None:
            try:
                data_imu1 = rospy.wait_for_message('/cf1/imu', Imu, timeout=5)
            except:
                rospy.loginfo("Crazyflie 1 imu not ready yet, retrying for getting robot imu")

        data_pose2 = None
        while data_pose2 is None:
            try:
                data_pose2 = rospy.wait_for_message('/cf2/local_position', GenericLogData, timeout=5)
            except:
                rospy.loginfo("Crazyflie 2 pose not ready yet, retrying for getting robot pose")

        data_imu2 = None
        while data_imu2 is None:
            try:
                data_imu2 = rospy.wait_for_message('/cf2/imu', Imu, timeout=5)
            except:
                rospy.loginfo("Crazyflie 2 imu not ready yet, retrying for getting robot imu")

        data_pose3 = None
        while data_pose3 is None:
            try:
                data_pose3 = rospy.wait_for_message('/cf3/local_position', GenericLogData, timeout=5)
            except:
                rospy.loginfo("Crazyflie 3 pose not ready yet, retrying for getting robot pose")

        data_imu3 = None
        while data_imu3 is None:
            try:
                data_imu3 = rospy.wait_for_message('/cf3/imu', Imu, timeout=5)
            except:
                rospy.loginfo("Crazyflie 3 imu not ready yet, retrying for getting robot imu")

        return data_pose1, data_imu1, data_pose2, data_imu2, data_pose3, data_imu3


    def calculate_dist_between_two_Points(self, p_init, p_end):
        a = np.array((p_init.x, p_init.y, p_init.z))
        b = np.array((p_end.x, p_end.y, p_end.z))

        dist = np.linalg.norm(a - b)

        return dist

    def check_topic_publishers_connection(self):

        rate = rospy.Rate(10)
        while (self.takeoff_pub1.get_num_connections() == 0):
            rate.sleep()

        while (self.vel_pub1.get_num_connections() == 0):
            rate.sleep()

        while (self.takeoff_pub2.get_num_connections() == 0):
            rate.sleep()

        while (self.vel_pub2.get_num_connections() == 0):
            rate.sleep()

        while (self.takeoff_pub3.get_num_connections() == 0):
            rate.sleep()

        while (self.vel_pub3.get_num_connections() == 0):
            rate.sleep()

    def reset_commands(self):
        vel_cmd = Hover()
        vel_cmd.vx = 0.0
        vel_cmd.vy = 0.0
        vel_cmd.zDistance = 0.0
        vel_cmd.yawrate = 0.0
        self.vel_pub.publish(vel_cmd)

    def takeoff_sequence(self, seconds_taking_off=3):

        takeoff_msg1 = FullState()
        takeoff_msg1.pose.position.z = 0.7
        takeoff_msg1.pose.position.y = 3
        takeoff_msg1.pose.position.x = 0

        takeoff_msg2 = FullState()
        takeoff_msg2.pose.position.z = 1
        takeoff_msg2.pose.position.y = 0
        takeoff_msg2.pose.position.x = 0

        takeoff_msg3 = FullState()
        takeoff_msg3.pose.position.z = 1.3
        takeoff_msg3.pose.position.y = -3
        takeoff_msg3.pose.position.x = 0

        rate = rospy.Rate(10)
        i = 0
        while i < 50:
            self.takeoff_pub1.publish(takeoff_msg1)
            self.takeoff_pub2.publish(takeoff_msg2)
            self.takeoff_pub3.publish(takeoff_msg3)
            i += 1
            rate.sleep()

    def into_pose(self, data_position):
        current_pose = Pose()
        current_pose.position.x = data_position.values[0]
        current_pose.position.y = data_position.values[1]
        current_pose.position.z = data_position.values[2]
        return current_pose

    def reward(self, data_position, data_imu, data_position2, data_position3):

        reward = 0

        current_pose = self.into_pose(data_position)
        current2_pose = self.into_pose(data_position2)
        current3_pose = self.into_pose(data_position3)

        mid = np.sum([data_position.values,data_position2.values, data_position3.values], axis = 0)/3

        mid_pose = Pose()
        mid_pose.position.x = mid[0]
        mid_pose.position.y = mid[1]
        mid_pose.position.z = mid[2]

        current_dist_12 = self.calculate_dist_between_two_Points(current_pose.position, current2_pose.position)
        current_dist_13 = self.calculate_dist_between_two_Points(current_pose.position, current3_pose.position)

        done = False

        positive_reward = False
        if positive_reward:
            done_bad = False
            done_crash = False

            # not too inclined and not too high
            euler = tf.transformations.euler_from_quaternion(
                [data_imu.orientation.x, data_imu.orientation.y, data_imu.orientation.z, data_imu.orientation.w])
            roll = euler[0]
            pitch = euler[1]
            yaw = euler[2]

            pitch_bad = not (-self.max_incl < pitch < self.max_incl)
            roll_bad = not (-self.max_incl < roll < self.max_incl)
            altitude_bad = data_position.values[2] > self.max_altitude

            if altitude_bad or pitch_bad or roll_bad:
                rospy.loginfo("(Drone flight status is wrong) >>> (" + str(altitude_bad) + "," + str(pitch_bad) + "," + str(
                    roll_bad) + ")")
                done_bad = True
                reward -= 200

            if not (current_dist_12<0.2 or current_dist_13<0.2):
                if current_dist_12 < 0.5 or current_dist_13 < 0.5:
                    #print(current_dist_12,current_dist_13)
                    reward -= -200
                    done_crash = True
            if (np.abs(current_dist_12 - self.formation_radius) < 0.1 and np.abs(current_dist_13 - self.formation_radius) < 0.1):
                reward = 200

            if not (current_dist_12 < 0.2 or current_dist_13 < 0.2):
                if self.calculate_dist_between_two_Points(current_pose.position, self.goal.position) < 0.2:
                    reward = 200

            #print(done_bad, done_crash)
            done = done_bad or done_crash


        if not (current_dist_12 < 0.2 or current_dist_13 < 0.2):
            reward -= (np.linalg.norm(current_dist_12 - self.formation_radius) + np.linalg.norm(current_dist_13 - self.formation_radius))
            #reward -= self.calculate_dist_between_two_Points(current_pose.position, self.goal.position)

            done1 = (np.abs(current_dist_12 - self.formation_radius) <= 0.1 and np.abs(current_dist_13 - self.formation_radius) <= 0.1)

            done = done1



        return reward, done