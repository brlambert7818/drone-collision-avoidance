#!/usr/bin/env python3

import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, DeleteModel, GetModelState
from gazebo_msgs.msg import ModelState
import roslaunch
import rospy
import time
from crazyflie_driver.msg import Hover, GenericLogData, Position, FullState
import numpy as np

class GazeboConnection():

    def __init__(self):
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.takeoff_pub = rospy.Publisher('/cf1/cmd_full_state', FullState, queue_size=1)


    def pauseSim(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException:
            print ("/gazebo/pause_physics service call failed")


    def unpauseSim(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException:
            print ("/gazebo/unpause_physics service call failed")


    def resetSim(self):
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            rospy.loginfo('Reset cf state')
            self.reset_proxy()
        except rospy.ServiceException:
            rospy.loginfo('Reset cf state failed')


    def resetSim_2(self):
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            rospy.loginfo('Reset cf state')
            self.reset_world()
        except rospy.ServiceException:
            rospy.loginfo('Reset cf state failed')


    def resetSim_new(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.reset_state(self.state_msg)
            # self.pause()
            print('reset new finished!')
        except rospy.ServiceException:
            print("/gazebo/reset_simulation service call failed")


    def reset_position(self):
        try:
            reset_msg = FullState()
            reset_msg.pose.position.x = 0
            reset_msg.pose.position.y = 0
            reset_msg.pose.position.z = 3

            rospy.loginfo("Go Home Start")
            rate = rospy.Rate(10)
            dist = np.inf
            while dist > 1:
                self.takeoff_pub.publish(reset_msg)
                cf_position = rospy.wait_for_message('/cf1/local_position', GenericLogData, timeout=5)
                dist = self.distance_between_points(cf_position.values[:3], (0, 0, 3))
                rate.sleep()

            rospy.loginfo("Go Home completed")

        except rospy.ServiceException:
            print("Go Home not working")
