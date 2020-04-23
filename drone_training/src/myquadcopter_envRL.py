#!/usr/bin/env python


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ROS packages required
import rospy
import rospkg


import gym
import rospy
import time
import numpy as np
import time
from gym import utils, spaces
from geometry_msgs.msg import Twist, Vector3Stamped, Pose
#from hector_uav_msgs.msg import Altimeter
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty as EmptyTopicMsg
from gym.utils import seeding
from gym.envs.registration import register
from gazebo_connection import GazeboConnection
from crazyflie_driver.msg import Hover, GenericLogData, Position, FullState

import random
from gym.spaces import Discrete, Box


from ray import tune
from ray.rllib.agents.pg.pg import PGTrainer
from ray.rllib.agents.pg.pg_policy import PGTFPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import try_import_tf





class Crazyflies(MultiAgentEnv):
    """Two-player environment for rock paper scissors.
    The observation is simply the last opponent action."""

    def __init__(self, _):
        self.action_space = Discrete(5)
        self.observation_space = Box(low=-np.inf, high=+np.inf, shape=(1,), dtype=np.float32)
        self.player1 = "player1"
        self.player2 = "player2"
        self.player3 = "player3"
        self.num_steps = 0

        # We assume that a ROS node has already been created
        # before initialising the environment

        self.takeoff_pub1 = rospy.Publisher('/cf1/cmd_full_state', FullState, queue_size=1)
        self.takeoff_pub2 = rospy.Publisher('/cf2/cmd_full_state', FullState, queue_size=1)
        self.takeoff_pub3 = rospy.Publisher('/cf3/cmd_full_state', FullState, queue_size=1)

        self.vel_pub1 = rospy.Publisher('/cf1/cmd_hover', Hover, queue_size=1)
        self.vel_pub2 = rospy.Publisher('/cf2/cmd_hover', Hover, queue_size=1)
        self.vel_pub3 = rospy.Publisher('/cf3/cmd_hover', Hover, queue_size=1)

        # gets training parameters from param server
        self.speed_value = rospy.get_param("drone1/speed_value")
        self.desired_pose1 = Pose()
        self.desired_pose2 = Pose()
        self.desired_pose3 = Pose()
        self.desired_pose1.position.z = rospy.get_param("/drone1/desired_pose/z")
        self.desired_pose1.position.x = rospy.get_param("/drone1/desired_pose/x")
        self.desired_pose1.position.y = rospy.get_param("/drone1/desired_pose/y")
        self.desired_pose2.position.z = rospy.get_param("/drone2/desired_pose/z")
        self.desired_pose2.position.x = rospy.get_param("/drone2/desired_pose/x")
        self.desired_pose2.position.y = rospy.get_param("/drone2/desired_pose/y")
        self.desired_pose3.position.z = rospy.get_param("/drone3/desired_pose/z")
        self.desired_pose3.position.x = rospy.get_param("/drone3/desired_pose/x")
        self.desired_pose3.position.y = rospy.get_param("/drone3/desired_pose/y")

        # in common for every drone
        self.running_step = rospy.get_param("/drone1/running_step")
        self.max_incl = rospy.get_param("/drone1/max_incl")
        self.max_altitude = rospy.get_param("/drone1/max_altitude")

        # stablishes connection with simulator
        self.gazebo = GazeboConnection()

        self.action_space = spaces.Discrete(5)  # Forward,Left,Right,Up,Down
        self.first = True

    def reset(self):
        self.num_steps = 0
        self.gazebo.unpauseSim()
        # 1st: resets the simulation to initial values
        if not self.first:
            self.gazebo.resetSim_porcodio3()
            #self.gazebo.resetSim_new5()
        self.first = False

        # 2nd: Unpauses simulation
        #self.gazebo.unpauseSim()

        # 3rd: resets the robot to initial conditions
        self.check_topic_publishers_connection() # SIMULATION RUNNING
        self.init_desired_pose() # SIMULATION RUNNING
        self.takeoff_sequence() # SIMULATION RUNNING

        # 4th: takes an observation of the initial condition of the robot
        data_pose1, data_imu1, data_pose2, data_imu2, data_pose3, data_imu3   = self.take_observation() # SIMULATION RUNNING

        observation_n = []

        observation1 = [data_pose1.values[0]] # SIMULATION RUNNING
        observation2 = [data_pose2.values[0]]  # SIMULATION RUNNING
        observation3 = [data_pose3.values[0]]  # SIMULATION RUNNING

        observation_n.append(observation1)
        observation_n.append(observation2)
        observation_n.append(observation3)

        # 5th: pauses simulation
        self.gazebo.pauseSim()

        return {
            self.player1: observation_n[0],
            self.player2: observation_n[1],
            self.player3: observation_n[2]
        }


    def step(self, action):

        # Given the action selected by the learning algorithm,
        # we perform the corresponding movement of the robot

        #vel_cmd1 = Hover()
        #vel_cmd2 = Hover()
        #vel_cmd3 = Hover()

        vel_cmd1 = self.decide_action(action[self.player1])
        vel_cmd2 = self.decide_action(action[self.player2])
        vel_cmd3 = self.decide_action(action[self.player3])

        # Then we send the command to the robot and let it go
        # for running_step seconds
        self.gazebo.unpauseSim()
        self.vel_pub1.publish(vel_cmd1)
        self.vel_pub2.publish(vel_cmd2)
        self.vel_pub3.publish(vel_cmd3)
        time.sleep(self.running_step)
        data_pose1, data_imu1, data_pose2, data_imu2, data_pose3, data_imu3  = self.take_observation()
        self.gazebo.pauseSim()

        # finally we get an evaluation based on what happened in the sim
        reward1, done1 = self.process_data(data_pose1, data_imu1, self.desired_pose1, self.best_dist1)
        reward2, done2 = self.process_data(data_pose2, data_imu2, self.desired_pose2, self.best_dist2)
        reward3, done3 = self.process_data(data_pose3, data_imu3, self.desired_pose3, self.best_dist3)

        done_n = done1 or done2 or done3

        # Promote going forwards instead if turning
        reward1 = self.reward_action(action[0], reward1)
        reward2 = self.reward_action(action[1], reward2)
        reward3 = self.reward_action(action[2], reward3)

        #reward_n = [reward1, reward2, reward3]

        state1 = [data_pose1.values[0]]
        state2 = [data_pose2.values[0]]
        state3 = [data_pose3.values[0]]
        #state_n = [state1, state2, state3]


        self.num_steps += 1

        obs = {
            self.player1: state1,
            self.player2: state2,
            self.player3: state3,
        }

        rew = {
            self.player1: reward1,
            self.player2: reward2,
            self.player3: reward3
        }

        done = {
            "__all__": done_n == True or self.num_steps >= 100,
        }

        return obs, rew, done, {}


    def decide_action(self,action):

        vel_cmd = Hover()
        if action == 0:  # FORWARD
            vel_cmd.vx = self.speed_value
            vel_cmd.yawrate = 0.0
            vel_cmd.zDistance = 1
        elif action == 1:  # LEFT
            vel_cmd.vy = self.speed_value
            vel_cmd.yawrate = 0.0
            vel_cmd.zDistance = 1
        elif action == 2:  # RIGHT
            vel_cmd.vy = -self.speed_value
            vel_cmd.yawrate = 0.0
            vel_cmd.zDistance = 1
        #elif action == 3:  # DIAGONALE SINISTRA
        #    vel_cmd.vx = self.speed_value
        #    vel_cmd.vy = self.speed_value
        #    vel_cmd.zDistance = 1
        #    vel_cmd.yawrate = 0.0
        #elif action == 4:  # DIAGONALE DESTRA
        #    vel_cmd.vx = self.speed_value
        #    vel_cmd.vy = -self.speed_value
        #    vel_cmd.zDistance = 1
        #    vel_cmd.yawrate = 0.0
        elif action == 3:  # INDIETRO
            vel_cmd.vx = -self.speed_value
            vel_cmd.vy = 0
            vel_cmd.zDistance = 1
            vel_cmd.yawrate = 0.0
        elif action == 4:  # STOP
            vel_cmd.vx = 0
            vel_cmd.vy = 0
            vel_cmd.zDistance = 1
            vel_cmd.yawrate = 0.0

        return vel_cmd

    def reward_action(self, action, reward):
        # Promote going forwards instead if turning
        if action == 0:
            reward += 100
        elif action == 1 or action == 2:
            reward -= 50
        elif action == 3:
            reward -= 150
        else:
            reward -= 50
        return reward



    def take_observation (self):
        data_pose1 = None
        while data_pose1 is None:
            try:
                data_pose1 = rospy.wait_for_message('/cf1/local_position', GenericLogData, timeout=5)
            except:
                rospy.loginfo("drone 1 pose not ready yet, retrying for getting robot pose")

        data_imu1 = None
        while data_imu1 is None:
            try:
                data_imu1 = rospy.wait_for_message('/cf1/imu', Imu, timeout=5)
            except:
                rospy.loginfo("drone 1 imu not ready yet, retrying for getting robot imu")


        data_pose2 = None
        while data_pose2 is None:
            try:
                data_pose2 = rospy.wait_for_message('/cf2/local_position', GenericLogData, timeout=5)
            except:
                rospy.loginfo("drone 2 pose not ready yet, retrying for getting robot pose")

        data_imu2 = None
        while data_imu2 is None:
            try:
                data_imu2 = rospy.wait_for_message('/cf2/imu', Imu, timeout=5)
            except:
                rospy.loginfo("drone 2 imu not ready yet, retrying for getting robot imu")

        data_pose3 = None
        while data_pose3 is None:
            try:
                data_pose3 = rospy.wait_for_message('/cf3/local_position', GenericLogData, timeout=5)
            except:
                rospy.loginfo("drone 3 pose not ready yet, retrying for getting robot pose")

        data_imu3 = None
        while data_imu3 is None:
            try:
                data_imu3 = rospy.wait_for_message('/cf3/imu', Imu, timeout=5)
            except:
                rospy.loginfo("drone 3 imu not ready yet, retrying for getting robot imu")

        return data_pose1, data_imu1, data_pose2, data_imu2, data_pose3, data_imu3

    def calculate_dist_between_two_Points(self,p_init,p_end):
        a = np.array((p_init.x ,p_init.y, p_init.z))
        b = np.array((p_end.x ,p_end.y, p_end.z))
        
        dist = np.linalg.norm(a-b)
        
        return dist



    def init_desired_pose(self):
        
        current_init_pose_pre1, imu1, current_init_pose_pre2, imu2, current_init_pose_pre3, imu3 = self.take_observation()
        current_init_pose1 = Pose()

        current_init_pose1.position.x = current_init_pose_pre1.values[0]
        current_init_pose1.position.y = current_init_pose_pre1.values[1]
        current_init_pose1.position.z = current_init_pose_pre1.values[2]

        self.best_dist1 = self.calculate_dist_between_two_Points(current_init_pose1.position, self.desired_pose1.position)

        current_init_pose2 = Pose()

        current_init_pose2.position.x = current_init_pose_pre2.values[0]
        current_init_pose2.position.y = current_init_pose_pre2.values[1]
        current_init_pose2.position.z = current_init_pose_pre2.values[2]

        self.best_dist2 = self.calculate_dist_between_two_Points(current_init_pose2.position,
                                                                 self.desired_pose2.position)

        current_init_pose3 = Pose()

        current_init_pose3.position.x = current_init_pose_pre3.values[0]
        current_init_pose3.position.y = current_init_pose_pre3.values[1]
        current_init_pose3.position.z = current_init_pose_pre3.values[2]

        self.best_dist3 = self.calculate_dist_between_two_Points(current_init_pose3.position,
                                                                 self.desired_pose3.position)

    def check_topic_publishers_connection(self):
        
        rate = rospy.Rate(10) # 10hz
        while(self.takeoff_pub1.get_num_connections() == 0):
            rospy.loginfo("No susbribers to Takeoff1 yet so we wait and try again")
            rate.sleep()
        rospy.loginfo("Takeoff1 Publisher Connected")

        while(self.vel_pub1.get_num_connections() == 0):
            rospy.loginfo("No susbribers to Cmd_vel1 yet so we wait and try again")
            rate.sleep()
        rospy.loginfo("Cmd_vel1 Publisher Connected")


        while (self.takeoff_pub2.get_num_connections() == 0):
            rospy.loginfo("No susbribers to Takeoff2 yet so we wait and try again")
            rate.sleep()
        rospy.loginfo("Takeoff2 Publisher Connected")

        while (self.vel_pub2.get_num_connections() == 0):
            rospy.loginfo("No susbribers to Cmd_vel2 yet so we wait and try again")
            rate.sleep()
        rospy.loginfo("Cmd_vel2 Publisher Connected")

        while (self.takeoff_pub3.get_num_connections() == 0):
            rospy.loginfo("No susbribers to Takeoff3 yet so we wait and try again")
            rate.sleep()
        rospy.loginfo("Takeoff Publisher3 Connected")

        while (self.vel_pub3.get_num_connections() == 0):
            rospy.loginfo("No susbribers to Cmd_vel3 yet so we wait and try again")
            rate.sleep()
        rospy.loginfo("Cmd_vel3 Publisher Connected")

    def reset_cmd_vel_commands(self):
        # We send an empty null Twist
        vel_cmd = Hover()
        #vel_cmd.linear.z = 0.0
        #vel_cmd.angular.z = 0.0
        vel_cmd.vx = 0.0
        vel_cmd.vy = 0.0
        vel_cmd.zDistance = 0.0
        vel_cmd.yawrate = 0.0
        self.vel_pub.publish(vel_cmd)
        rospy.loginfo("Resetted commands")



    def takeoff_sequence(self, seconds_taking_off=3):
        # Before taking off be sure that cmd_vel value there is is null to avoid drifts
        #self.reset_cmd_vel_commands()

        takeoff_msg1 = FullState()
        takeoff_msg1.pose.position.z = 1
        takeoff_msg1.pose.position.y = 1
        takeoff_msg1.pose.position.x = 0

        takeoff_msg2 = FullState()
        takeoff_msg2.pose.position.z = 1
        takeoff_msg2.pose.position.y = 0
        takeoff_msg2.pose.position.x = -1

        takeoff_msg3 = FullState()
        takeoff_msg3.pose.position.z = 1
        takeoff_msg3.pose.position.y = -1
        takeoff_msg3.pose.position.x = 0

        rospy.loginfo( "Taking-Off Start")
        rate = rospy.Rate(10)
        i = 0
        while i < 50:
            self.takeoff_pub1.publish(takeoff_msg1)
            self.takeoff_pub2.publish(takeoff_msg2)
            self.takeoff_pub3.publish(takeoff_msg3)
            i += 1
            rate.sleep()
        rospy.loginfo( "Taking-Off sequence completed")
        

    def improved_distance_reward(self, current_pose_pre, desired_pose, best_dist):
        current_pose = Pose()
        current_pose.position.x = current_pose_pre.values[0]
        current_pose.position.y = current_pose_pre.values[1]
        current_pose.position.z = current_pose_pre.values[2]
        current_dist = self.calculate_dist_between_two_Points(current_pose.position, desired_pose.position)
        #rospy.loginfo("Calculated Distance = "+str(current_dist))
        
        if current_dist < best_dist:
            reward = 100
            best_dist = current_dist
        elif current_dist == best_dist:
            reward = 0
        else:
            reward = -100
            #print "Made Distance bigger= "+str(self.best_dist)
        
        return reward
        
    def process_data(self, data_position, data_imu, desired_pose, best_dist):

        done = False
        
        euler = tf.transformations.euler_from_quaternion([data_imu.orientation.x,data_imu.orientation.y,data_imu.orientation.z,data_imu.orientation.w])
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]

        pitch_bad = not(-self.max_incl < pitch < self.max_incl)
        roll_bad = not(-self.max_incl < roll < self.max_incl)
        altitude_bad = data_position.values[2] > self.max_altitude

        if altitude_bad or pitch_bad or roll_bad:
            rospy.loginfo ("(Drone flight status is wrong) >>> ("+str(altitude_bad)+","+str(pitch_bad)+","+str(roll_bad)+")")
            done = True
            reward = -200
        else:
            reward = self.improved_distance_reward(data_position, desired_pose, best_dist)

        return reward,done


if __name__ == '__main__':

    rospy.init_node('drone_gym', anonymous=True)
    #tf = try_import_tf()

    rospy.loginfo("Gym environment done")
    observation_space = Box(low=-np.inf, high=+np.inf, shape=(1,), dtype=np.float32)

    def select_policy(agent_id):
        if agent_id == "player1":
            return "player1"
        elif agent_id == "player2":
            return "player2"
        else:
            return "player3"


    tune.run(
        "PG",
        stop={"timesteps_total": 400000},
        config={
            "env": Crazyflies,
            "gamma": 0.9,
            "num_workers": 1,
            "num_envs_per_worker": 1,
            "ignore_worker_failures": True,
            "sample_batch_size": 10,
            "train_batch_size": 200,
            "multiagent": {
                "policies_to_train": ["player1", "player2", "player3" ],
                "policies": {
                    "player1": (None, observation_space, Discrete(3), {
                        "model": {
                            "use_lstm": False
                        }
                    }),
                    "player2": (None, observation_space, Discrete(3), {
                        "model": {
                            "use_lstm": False
                        }
                    }),
                    "player3": (None, observation_space, Discrete(3), {
                        "model": {
                            "use_lstm": False
                        }
                    }),
                },
                "policy_mapping_fn": tune.function(select_policy),
            },
        })