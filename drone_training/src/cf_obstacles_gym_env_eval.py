#!/usr/bin/env python3

import gym
import rospy
import time
import numpy as np
import time
import math
from gym import utils, spaces
from geometry_msgs.msg import Twist, Vector3Stamped, Pose
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty as EmptyTopicMsg
from std_srvs.srv import Empty
from gym.utils import seeding
from gym.envs.registration import register
from crazyflie_driver.msg import Hover, GenericLogData, Position, FullState
import signal 
import os
import subprocess
import crazyflie

reg = register(
    id='CrazyflieObstacleEval-v0',
    entry_point='cf_obstacles_gym_env_eval:CrazyflieObstacleEnvEval',
    )

class CrazyflieObstacleEnvEval(gym.Env):

    def __init__(self, n_obstacles, avoidance_method):
        super(CrazyflieObstacleEnv, self).__init__()

        self.n_obstacles = n_obstacles
        self.avoidance_method = self.set_avoidance_method(avoidance_method)
        self.cfs = np.empty(n_obstacles + 1, dtype=object) 
        self.steps_since_avoided = np.zeros(n_obstacles)

        # Low-level Crazyflie control methods
        self.full_state_pub = rospy.Publisher('/cf1/cmd_full_state', FullState, queue_size=1)
        self.position_pub = rospy.Publisher('/cf1/cmd_position', Position, queue_size=1)
        self.hover_pub = rospy.Publisher('/cf1/cmd_hover', Hover, queue_size=1)
        self.vel_pub = rospy.Publisher('/cf1/cmd_vel', Twist, queue_size=1)
        
        self.hover_pubs = np.empty(n_obstacles+1, dtype=object)
        for i in range(self.n_obstacles+1):
            self.hover_pubs[i] = rospy.Publisher('/cf' + str(i+1) + '/cmd_hover', Hover, queue_size=1)

        self.goal_position = np.array((2.5, 2.5, 4))

        # establishes connection with simulator
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.gazebo_process, self.cf_process = self.launch_sim()

        # Gym spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(6 + (3*n_obstacles),), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(6,), dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)
        self.steps = 0


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):

        self.unpauseSim()

        self.steps = 0
        self.steps_since_avoided = np.zeros(self.n_obstacles)
        reset_positions = self.random_position(-4, 5, 1, 10, self.n_obstacles + 1)

        # for eval
        # reset_positions[1] = self.goal_position

        print('Start Reset')

        # Connect to Crazyflie and enable high-level control
        for i in range(self.n_obstacles + 1):
            self.cfs[i] = crazyflie.Crazyflie('cf' + str(i+1), '/cf' + str(i+1))
            self.cfs[i].setParam("commander/enHighLevel", 1)
        time.sleep(2)
            
        for i in range(self.n_obstacles + 1):
            self.cfs[i].takeoff(targetHeight = reset_positions[i][2], duration = 4)
        time.sleep(4)

        for i in range(self.n_obstacles + 1):
            self.cfs[i].goTo(goal=[reset_positions[i][0], reset_positions[i][1], reset_positions[i][2]], yaw=0.0, duration=4)
        time.sleep(4)

        # cf1 reset
        ############################

        # cf_position = self.get_position(1)
        # action_msg = Hover()
        # action_msg.vx = cf_position[0] 
        # action_msg.vy = cf_position[1] 
        # action_msg.zDistance = reset_positions[0][2]
        # action_msg.yawrate = 0 

        # while abs(self.get_position(1)[2] - reset_positions[0][2]) > 1:
        #     self.hover_pub.publish(action_msg)
        #     time.sleep(0.3)

        #     # check if drone flipped during the reset
        #     roll = self.get_pose(1)[3]
        #     if abs(roll) >= 60:
        #         self.kill_sim()
        #         time.sleep(20)
        #         self.gazebo_process, self.cf_process = self.launch_sim()

        # action_msg = Position()
        # action_msg.x = reset_positions[0][0] 
        # action_msg.y = reset_positions[0][1]
        # action_msg.z = reset_positions[0][2] 
        # action_msg.yaw = 0 

        # while self.distance_between_points(self.get_position(1), reset_positions[0]) > 1:
        #     self.position_pub.publish(action_msg)
        #     time.sleep(0.3)

        #     # check if drone flipped during the reset
        #     roll = self.get_pose(1)[3]
        #     if abs(roll) >= 60:
        #         self.kill_sim()
        #         time.sleep(20)
        #         self.gazebo_process, self.cf_process = self.launch_sim()

        ############################

        print('End Reset')

        self.pauseSim()
        return self.get_observation()


    def get_observation(self):

        observation = np.array([])
        for i in range(1, self.n_obstacles+2):
            pose = None
            while pose is None:
                try:
                    pose = rospy.wait_for_message('/cf' + str(i) + '/local_position', GenericLogData, timeout=10)
                except:
                    rospy.loginfo("Crazyflie 1 pose not ready yet, retrying for getting robot pose")
                    self.kill_sim()
                    time.sleep(20)
                    self.gazebo_process, self.cf_process = self.launch_sim()

            # get x,y,z position
            position = np.array(pose.values[:3])
            observation = np.append(observation, position)

            if i == 1:
                # get roll, pitch, and yaw Euler angles
                roll_pitch_yaw = np.array(pose.values[3:])
                observation = np.append(observation, roll_pitch_yaw)

        # ob_vel = np.array([])
        # for i in range(2, self.n_obstacles+2):
        #     # get angular velocities and linear accelerations 
        #     imu = None
        #     while imu is None:
        #         try:
        #             imu = rospy.wait_for_message('/cf' + str(i) + '/imu', Imu, timeout=5)
        #         except:
        #             rospy.loginfo("Crazyflie 1 imu not ready yet, retrying for getting robot imu")

        #     linear_acceleration = np.array((imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z))
        #     ob_vel.append(linear_acceleration)

        # return np.concatenate((position, angular_velocity, linear_acceleration, roll_pitch_yaw))

        return observation


    def step(self, action):

        avoided_collision = False 
        needs_to_avoid = False
        
        if self.distance_between_points(self.get_position(1), self.get_position(2)) < 1:
            needs_to_avoid = True

        # Perform the action
        self.steps += 1
        action_msg = self.process_action(action)

        self.unpauseSim()
        self.hover_pub.publish(action_msg)
        time.sleep(0.3)
        self.pauseSim()

        if needs_to_avoid:
            avoided_collision = self.distance_between_points(self.get_position(1), self.get_position(2)) > 0.5 

        self.unpauseSim()
        # Move the obstacles
        self.move_obstacles()
        time.sleep(0.3)
        self.pauseSim()

        # if self.distance_between_points(self.get_position(1), self.get_position(2)) < 1:
        #     needs_to_avoid = True

        # Get next observation
        observation = self.get_observation()

        # Check if flipped
        is_flipped = False
        for i in range(1, self.n_obstacles+2):
            roll = self.get_pose(i)[3]
            if abs(roll) >= 60:
                is_flipped = True
                print('FLIPPED....')
                break

        # Check for a collision
        # is_collision = False
        # if needs_to_avoid and not avoided_collision:
        #     is_collision = True

        is_collision = False
        for i in range(2, self.n_obstacles+2):
            if self.distance_between_points(self.get_position(1), self.get_position(i)) < 0.5:
                is_collision = True
                break

        reached_goal = False
        if self.distance_between_points(self.get_position(1), self.goal_position) < 1:
            reached_goal = True
        
        max_steps = False
        if self.steps == 256:
            max_steps = True

        reward, is_terminal = self.reward(observation, is_flipped, is_collision, reached_goal, max_steps) 
        
        # Still need ob-on-ob repulsion even if main cf will not use collision heuristic
        # start_cf = 1 if self.avoidance_method != 'None' else 2

        # # Crazyflies being repelled
        # for i in range(start_cf, self.n_obstacles+1):
        #     positions = np.zeros(self.n_obstacles, dtype=object)
        #     dist_from_ob = np.repeat(np.inf, self.n_obstacles)

        #     # Crazyflies doing the repelling
        #     for j in range(i+1, self.n_obstacles+2):
        #         cf_position = self.get_position(i) 
        #         ob_position = self.get_position(j)
        #         positions[j-2] = ob_position
        #         dist_from_ob[j-2] = self.distance_between_points(cf_position, ob_position)

        #     # Main cf repelled by closest obstacle 
        #     min_dist = np.min(dist_from_ob)
        #     if min_dist < 1:
        #         min_index = np.argmin(dist_from_ob)
        #         cf_position = self.get_position(i)

        #         # Always use repulsion for ob-on-ob avoidance
        #         if i != 1:
        #             self.repel(cf_position, positions[min_index], cf_id=i, ob_id=min_index+2)

        #         # Use chosen avoidance method for cf1
        #         elif self.avoidance_method == 'Heuristic':
        #             self.unpauseSim()
        #             self.repel(cf_position, positions[min_index], cf_id=i, ob_id=min_index+2)

        #             action_msg = Hover()
        #             action_msg.vx = cf_position[0] 
        #             action_msg.vy = cf_position[1] 
        #             action_msg.zDistance = cf_position[2]
        #             action_msg.yawrate = 0 

        #             for _ in range(2):
        #                 self.hover_pub.publish(action_msg)
        #                 time.sleep(0.3)

        #             self.pauseSim()

        #             needs_to_avoid = True
        #             avoided_collision = True
        #         elif self.avoidance_method == 'RL Separate':
        #             pass
        #         elif self.avoidance_method == 'RL Combined':
        #             pass
        
        # observation = self.get_observation()

        # if needs_to_avoid:
        #     avoided_collision = self.distance_between_points(self.get_position(1), self.get_position(2)) > 0.5 

        # is_collision = False
        # if needs_to_avoid and not avoided_collision:
        #     is_collision = True

        # reward, is_terminal = self.reward(observation, is_flipped, is_collision, reached_goal, max_steps) 

        # Restart simulation if drone has flipped
        if is_flipped:
            self.kill_sim()
            time.sleep(25)
            self.gazebo_process, self.cf_process = self.launch_sim()
        
        return observation, reward, is_terminal, {"needs_to_avoid": needs_to_avoid,
                                                  "avoided_collision": avoided_collision, 
                                                  "reached_goal": reached_goal,
                                                  "flipped": is_flipped, 
                                                  "exceeded_steps": max_steps
                                                  }
    

    def reward(self, observation, is_flipped, is_collision, reached_goal, max_steps):

        # dist_to_goal = self.distance_between_points(observation[:3], self.goal_position)
        dist_to_goal = self.distance_between_points(self.get_position(1), self.goal_position)
        reward = 0
        is_terminal = False

        if is_collision:
            print('CRASHED....')
            reward -= 100 
            is_terminal = True

        # Reached goal
        # if dist_to_goal < 1:
        if reached_goal:
            reward += 100 
            is_terminal = True 
            print('REACHED GOAL.....')
        else:
            # if self.steps == 256:
            if max_steps:
                is_terminal = True
            # Penalize based on distance to goal
            reward -= dist_to_goal / 500
            if is_flipped:
                is_terminal = True

        return reward, is_terminal


    def close(self):
        self.kill_sim()


################################################################################
#                           Helper Functions
################################################################################


    def set_avoidance_method(self, avoidance_method):
        methods = ['Heuristic', 'RL Separate', 'RL Combined', 'None']
        if avoidance_method in methods:
            return avoidance_method 
        else:
            raise Exception('Invalid collision avoidance method chose. Please choose from the following: \n' + '\n'.join(methods))
            


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


    def repel(self, cf_position, ob_position, cf_id, ob_id):

        if cf_id == 1:
            self.steps_since_avoided[ob_id-2] = 1 

        # Get x,y velocities from tangential repulsion
        tan_angle = math.atan2(cf_position[1] - ob_position[1],
                                    cf_position[0] - ob_position[0])
        tan_angle = (tan_angle + 2*math.pi) % (2*math.pi) 
        vel_xy = np.array([math.cos(tan_angle), math.sin(tan_angle)])

        # Convert to max velocities for avoidance
        for i in range(2):
            if vel_xy[i] > 0:
                vel_xy[i] = 1 if cf_id == 1 else 0.5
            elif vel_xy[i] < 0:
                vel_xy[i] = -1 if cf_id == 1 else -0.5
        
        # Get vertical avoidance
        z_increase = cf_position[2] >= ob_position[2]

        action_msg = Hover()
        action_msg.vx = vel_xy[0] 
        action_msg.vy = vel_xy[1] 
        action_msg.zDistance = cf_position[2] 
        action_msg.yawrate = 0 

        # self.unpauseSim()
        for _ in range(4):
            cf_position = self.get_position(1)

            if cf_id == 1:
                # Bounce drone off right wall
                if cf_position[0] >= 9.5 and action_msg.vx > 0:
                    action_msg.vx = 0
                    cf_position = self.get_position(1)
                # Bounce drone off left wall
                elif cf_position[0] <= -9.5 and action_msg.vx < 0:
                    action_msg.vx = 0
                    cf_position = self.get_position(1)

                # Bounce drone off back wall
                if cf_position[1] >= 9.5 and action_msg.vy > 0:
                    action_msg.vy = 0
                    cf_position = self.get_position(1)
                # Bounce drone off front wall
                elif cf_position[1] <= -9.5 and action_msg.vy < 0:
                    action_msg.vy = 0
                    cf_position = self.get_position(1)

            # Move cf upwards or downwards based on relative position of obstacle
            if z_increase:
                action_msg.zDistance += 0.1
            else:
                action_msg.zDistance -= 0.1
            action_msg.zDistance = np.clip(action_msg.zDistance, 0.5, 9.5)

            print('Repel %i from %i' % (cf_id, ob_id))


            self.hover_pubs[cf_id-1].publish(action_msg)
            time.sleep(0.3)
        
        self.pauseSim()


    def get_position(self, cf_id):
        pose = None
        while pose is None:
            try:
                pose = rospy.wait_for_message('/cf' + str(cf_id) + '/local_position', GenericLogData, timeout=5)
            except:
                rospy.loginfo("Crazyflie pose not ready yet, retrying for getting robot pose")
                self.kill_sim()
                time.sleep(20)
                self.gazebo_process, self.cf_process = self.launch_sim()

        # get x,y,z position
        position = np.array(pose.values[:3])
        return position


    def get_pose(self, cf_id):
        pose = None
        while pose is None:
            try:
                pose = rospy.wait_for_message('/cf' + str(cf_id) + '/local_position', GenericLogData, timeout=5)
            except:
                rospy.loginfo("Crazyflie pose not ready yet, retrying for getting robot pose")
                self.kill_sim()
                time.sleep(20)
                self.gazebo_process, self.cf_process = self.launch_sim()

        return np.array(pose.values) 


    def get_velocities(self, cf_id):
        imu = None
        while imu is None:
            try:
                imu = rospy.wait_for_message('/cf' + str(cf_id) + '/imu', Imu, timeout=5)
            except:
                rospy.loginfo("Crazyflie 1 imu not ready yet, retrying for getting robot imu")
                self.kill_sim()
                time.sleep(20)
                self.gazebo_process, self.cf_process = self.launch_sim()

        angular_velocity = np.array((imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z))
        linear_acceleration = np.array((imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z))
        return np.concatenate((angular_velocity, linear_acceleration))


    def launch_sim(self):

        """ Executes bash commands to launch the Gazebo simulation, spawn a Crazyflie 
        UAV, and create a controller for the Crazyflie.

        Returns:
            bash process: Process corresponding to the Gazebo simulation
            bash process: Process corresponding to the Crazyflie model and controller 
        """
        rospy.loginfo('LAUNCH SIM')
        
        launch_gazebo_cmd = ''
        if self.n_obstacles > 0:
            launch_gazebo_cmd = 'roslaunch crazyflie_gazebo multiple_cf_sim_' + str(self.n_obstacles+1) + '.launch'
        else:
            launch_gazebo_cmd = 'roslaunch crazyflie_gazebo crazyflie_sim.launch'
        gazebo_process = subprocess.Popen(launch_gazebo_cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        time.sleep(5)

        cf_gazebo_path = '/home/brian/catkin_ws/src/sim_cf/crazyflie_gazebo/scripts'
        launch_controller_cmd = './run_cfs.sh ' + str(self.n_obstacles + 1)
        cf_process = subprocess.Popen(launch_controller_cmd, stdout=subprocess.PIPE, cwd=cf_gazebo_path, shell=True, preexec_fn=os.setsid)
        self.unpauseSim()
        time.sleep(5)

        return gazebo_process, cf_process


    def kill_sim(self):
        """ Terminates the Gazeo and Crazyflie processes.
        """
        rospy.loginfo('KILL SIM')
        os.killpg(os.getpgid(self.cf_process.pid), signal.SIGTERM)
        os.killpg(os.getpgid(self.gazebo_process.pid), signal.SIGTERM)


    def random_position(self, xy_min, xy_max, z_min, z_max, n_positions):
        while True:
            try:
                xy = np.random.randint(xy_min, xy_max, size=(n_positions, 2))
                z = np.random.randint(z_min, z_max, size=(n_positions, 1))
                xyz = np.append(xy, z, axis=1)
                if not np.array_equal(xyz[0], self.goal_position): return xyz
            except:
                pass


    def move_obstacles(self):
        # Get primary agent position
        cf_position = self.get_position(1)

        for i in range(2, self.n_obstacles + 2):

            # Add noise to avoid obstacle always being on the tail of the agent
            # target_position = cf_position + np.random.normal(-0.5, 0.5, size = 3)
            target_position = cf_position
            # Obstacle must wait 5 steps to move after it has been avoided
            if self.steps_since_avoided[i-2] == 0:
                self.cfs[i-1].goTo(goal=[target_position[0]-0.1, target_position[1], np.clip(target_position[2], 0.25, 9.5)], yaw=0.0, duration=3)

        # Update steps since avoidance
        for i in range(self.n_obstacles):
            if self.steps_since_avoided[i] > 0:
                if self.steps_since_avoided[i] == 1:
                    self.steps_since_avoided[i] = 0
                else:
                    self.steps_since_avoided[i] += 1


    def process_action(self, action):

        action[0] = self.unnormalize(action[0], -0.4, 0.4)
        action[1] = self.unnormalize(action[1], -0.4, 0.4)
        action[2] = self.unnormalize(action[2], 0.25, 9.75)
        # action[3] = self.unnormalize(action[3], -200, 200)
        
        cf_posititon = self.get_position(1)
        
        # Bounce drone off right wall
        if cf_posititon[0] >= 4.5 and action[0] > 0:
            action[0] = 0
            cf_posititon = self.get_position(1)
        # Bounce drone off left wall
        elif cf_posititon[0] <= -4.5 and action[0] < 0:
            action[0] = 0
            cf_posititon = self.get_position(1)
        # Bounce drone off back wall
        if cf_posititon[1] >= 4.5 and action[1] > 0:
            action[1] = 0
            cf_posititon = self.get_position(1)
        # Bounce drone off front wall
        elif cf_posititon[1] <= -4.5 and action[1] < 0:
            action[1] = 0
            cf_posititon = self.get_position(1)

        # Option 1: Hovering movements 
        action_msg = Hover()
        action_msg.vx = action[0] 
        action_msg.vy = action[1] 
        action_msg.zDistance = action[2]
        # action_msg.yawrate = action[3] 
        action_msg.yawrate = 0 

        # Option 2: Velocity movements 
        # action_msg = Twist()
        # action_msg.linear.x = action[0]
        # action_msg.linear.y = action[1]
        # action_msg.linear.z = action[2]
        # action_msg.angular.z = 0 

        # Option 3: Positon movements 
        # action_msg = Position()
        # action_msg.x = action[0] 
        # action_msg.y = action[1]
        # action_msg.z = action[2] 
        # action_msg.yaw = 0 

        return action_msg 


    def unnormalize(self, x_norn, x_min, x_max):
        return (x_max - x_min) * ((x_norn / 2) + 0.5) + x_min


    def distance_between_points(self, point_a, point_b):
        return np.linalg.norm(point_a - point_b)
