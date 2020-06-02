#!/usr/bin/env python

import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState, DeleteModel
from gazebo_msgs.msg import ModelState
import roslaunch
import rospy
import time
from crazyflie_driver.msg import Hover, GenericLogData, Position, FullState

class GazeboConnection():

    def __init__(self):
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.reset_diocane = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        self.state_msg = ModelState()
        self.state_msg.model_name = 'cf1'
        self.state_msg.pose.position.x = 0
        self.state_msg.pose.position.y = 0
        self.state_msg.pose.position.z = 0.03
        self.state_msg.pose.orientation.x = 0
        self.state_msg.pose.orientation.y = 0
        self.state_msg.pose.orientation.z = 0
        self.state_msg.pose.orientation.w = 1
        self.state_msg.twist.linear.x = 0
        self.state_msg.twist.linear.y = 0
        self.state_msg.twist.linear.z = 0
        self.state_msg.twist.angular.x = 0
        self.state_msg.twist.angular.y = 0
        self.state_msg.twist.angular.z = 0

        self.takeoff_pub = rospy.Publisher('/cf1/cmd_full_state', FullState, queue_size=1)
        self.takeoff_pub1 = rospy.Publisher('/cf1/cmd_full_state', FullState, queue_size=1)
        self.takeoff_pub2 = rospy.Publisher('/cf2/cmd_full_state', FullState, queue_size=1)
        self.takeoff_pub3 = rospy.Publisher('/cf3/cmd_full_state', FullState, queue_size=1)


    def pauseSim(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")


    def unpauseSim(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException, e:
            print ("/gazebo/unpause_physics service call failed")


    def resetSim(self):
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except rospy.ServiceException, e:
            print ("/gazebo/reset_simulation service call failed")


    def resetSim_new(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.reset_state(self.state_msg)
            self.pause()
        except rospy.ServiceException, e:
            print("/gazebo/reset_simulation service call failed")


    def resetSim_new2(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            rase = rospy.Rate(10)
            self.reset_diocane('cf1')
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)
            launch = roslaunch.parent.ROSLaunchParent(uuid,
                                                      ["/home/alberto/catkin_ws5/src/sim_cf/crazyflie_gazebo/launch/spawn_mav2.launch"])
            launch.start()
            rospy.loginfo("Spawn started")
            asd = 0
            while asd < 50:
                rase.sleep()
                asd += 1
            rospy.loginfo("Spawn ended")

            # 3 seconds later
            #launch.shutdown()

        except rospy.ServiceException, e:
            print("/gazebo/reset_simulation service call failed")


    def resetSim_new5(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:

            rase = rospy.Rate(10)
            self.reset_diocane('cf1')
            self.reset_diocane('cf2')
            self.reset_diocane('cf3')
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)
            launch = roslaunch.parent.ROSLaunchParent(uuid,
                                                      [
                                                          "/home/alberto/catkin_ws5/src/sim_cf/crazyflie_gazebo/launch/3_cf_sim_reset.launch"])
            launch.start()
            rospy.loginfo("Spawn started")
            asd = 0
            while asd < 50:
                rase.sleep()
                asd += 1
            rospy.loginfo("Spawn ended")

            # 3 seconds later
            # launch.shutdown()

        except rospy.ServiceException, e:
            print("/gazebo/reset_simulation service call failed")


    def resetSim_new3(self):
        try:
            takeoff_msg = Position()
            takeoff_msg.z = 1
            takeoff_msg.y = 0
            takeoff_msg.x = 0
            rospy.loginfo("Go Home Start")
            i = 0
            while i<100:
                self.takeoff_pub.publish(takeoff_msg)
                i += 1

            rospy.loginfo("Go Home completed")
        except rospy.ServiceException, e:
            print("/gazebo/reset_simulation service call failed")


    def resetSim_definitive(self):
        try:
            rate = rospy.Rate(10)
            takeoff_msg = FullState()
            takeoff_msg.pose.position.z = 5
            takeoff_msg.pose.position.y = 0
            takeoff_msg.pose.position.x = 0
            rospy.loginfo("Go Home Start")
            i = 0
            while i < 200:
                self.takeoff_pub.publish(takeoff_msg)
                i += 1
                rate.sleep()

            rospy.loginfo("Go Home completed")
        except rospy.ServiceException, e:
            print("Go Home not owrking")


    def resetSim_definitive3(self):
        try:
            rate = rospy.Rate(10)
            takeoff_msg1 = FullState()
            takeoff_msg1.pose.position.z = 0.5
            takeoff_msg1.pose.position.y = 3
            takeoff_msg1.pose.position.x = 0

            takeoff_msg2 = FullState()
            takeoff_msg2.pose.position.z = 1
            takeoff_msg2.pose.position.y = 0
            takeoff_msg2.pose.position.x = 0

            takeoff_msg3 = FullState()
            takeoff_msg3.pose.position.z = 2
            takeoff_msg3.pose.position.y = -3
            takeoff_msg3.pose.position.x = 0

            #rospy.loginfo("Go Home Start")
            i = 0
            while i < 100:
                self.takeoff_pub1.publish(takeoff_msg1)
                self.takeoff_pub2.publish(takeoff_msg2)
                self.takeoff_pub3.publish(takeoff_msg3)
                i += 1
                rate.sleep()

            #rospy.loginfo("Go Home completed")
        except rospy.ServiceException, e:
            print("Go Home not working")


    def resetSim_definitive4(self):
        try:
            rate = rospy.Rate(10)
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

            #rospy.loginfo("Go Home Start")
            i = 0
            while i < 90:
                self.takeoff_pub1.publish(takeoff_msg1)
                self.takeoff_pub2.publish(takeoff_msg2)
                self.takeoff_pub3.publish(takeoff_msg3)
                i += 1
                rate.sleep()

            #rospy.loginfo("Go Home completed")
        except rospy.ServiceException, e:
            print("Go Home not working")
