#!/usr/bin/env python

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
