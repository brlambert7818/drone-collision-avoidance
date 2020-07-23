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
    id='CrazyflieObstacle-v0',
    entry_point='cf_obstacle_gym_env:CrazyflieObstacleEnv',
    )

class CrazyflieObstacleEnv(gym.Env):

    def __init__(self, n_obstacles, avoidance_method):
        super(CrazyflieObstacleEnv, self).__init__()

        self.n_obstacles = n_obstacles
        self.avoidance_method = self.set_avoidance_method(avoidance_method)
        self.cfs = np.empty(n_obstacles + 1, dtype=object) 
        self.steps_since_avoided = np.zeros(n_obstacles)

        # Low-level Crazyflie control methods
        self.full_state_pub = rospy.Publisher('/cf1/cmd_full_state', FullState, queue_size=1)
        self.position_pub = rospy.Publisher('/cf2/cmd_position', Position, queue_size=1)
        self.hover_pub = rospy.Publisher('/cf1/cmd_hover', Hover, queue_size=1)
        self.vel_pub = rospy.Publisher('/cf1/cmd_vel', Twist, queue_size=1)
        
        self.hover_pubs = np.empty(n_obstacles+1, dtype=object)
        for i in range(self.n_obstacles+1):
            self.hover_pubs[i] = rospy.Publisher('/cf' + str(i+1) + '/cmd_hover', Hover, queue_size=1)

        self.goal_position = np.array((2.5, 2.5, 5))

        # establishes connection with simulator
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.gazebo_process, self.cf_process = self.launch_sim()

        # Gym spaces
        self.action_space = spaces.Box(low=np.array([-0.4, -0.4, 0.5]), high=np.array([0.4, 0.4, 9.5]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(12,), dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)
        self.steps = 0


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

        self.steps = 0
        self.steps_since_avoided = np.zeros(self.n_obstacles)

        # Connect to Crazyflie and enable high-level control
        for i in range(self.n_obstacles + 1):
            self.cfs[i] = crazyflie.Crazyflie('cf' + str(i+1), '/cf' + str(i+1))
            self.cfs[i].setParam("commander/enHighLevel", 1)

        reset_positions = self.random_position(-4, 5, 1, 10, self.n_obstacles + 1)
            
        for i in range(self.n_obstacles + 1):
            self.cfs[i].takeoff(targetHeight = reset_positions[i][2], duration = 4)
        time.sleep(4)
        for i in range(self.n_obstacles + 1):
            self.cfs[i].goTo(goal=[reset_positions[i][0], reset_positions[i][1], reset_positions[i][2]], yaw=0.0, duration=4)
        time.sleep(4)

        return self.get_observation()


    def get_observation(self):
        """ Returns the current drone state consisting of the following: (x, y, z)
        positions, (x, y, z) angular velocities, (x, y, z) linear accelerations, 
        and (roll, pitch, yaw) Euler angles in degrees. 

        Returns:
            ndarray (dtype=float, ndim=1): Array containing the current state observation.  
        """

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

        # get angular velocities and linear accelerations
        imu = None
        while imu is None:
            try:
                imu = rospy.wait_for_message('/cf1/imu', Imu, timeout=5)
            except:
                rospy.loginfo("Crazyflie 1 imu not ready yet, retrying for getting robot imu")

        angular_velocity = np.array((imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z))
        linear_acceleration = np.array((imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z))

        return np.concatenate((position, angular_velocity, linear_acceleration, roll_pitch_yaw))


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

        # Perform the action
        self.steps += 1
        action_msg = self.process_action(action)
        self.hover_pub.publish(action_msg)
        time.sleep(0.3)
        self.move_obstacles()
        time.sleep(0.3)

        # Get next observation
        observation = self.get_observation()

        # Check if flipped
        is_flipped = False
        if abs(observation[-3]) >= 60:
            is_flipped = True
            print('FLIPPED....')
        reward, is_terminal = self.reward(observation, is_flipped) 
        
        if self.avoidance_method != 'None':
            # dist_from_ob = np.repeat(np.inf, self.n_obstacles)
            # positions = np.zeros(self.n_obstacles, dtype=object)
            # Check if collision is imminent 
            for i in range(2, self.n_obstacles + 2):
                cf_position = observation[:3]
                ob_position = self.get_position(i)
                # positions[i-2] = ob_position
                # dist_from_ob[i-2] = self.distance_between_points(cf_position, ob_position)

                if self.distance_between_points(cf_position, ob_position) < 1:
                    if self.avoidance_method == 'Heuristic':
                        self.repel(cf_position, ob_position, cf_id=1, ob_id=i)
                    # Only repel from the first obstacle. Better implementation would
                    # be to react to the closest. Code below does this but does not work yet
                    break 
            

            # TO DO: 
            #   - change angle of obstacles after collision so 2 doesn't come 
            #     back and collide with 3 right away
            #   - reduce iters and/or vel of obstacle-on-obstacle repulsion 
            #   - figure out how to not hard code obstacle repulsion
            cf2_pos = self.get_position(2)
            cf3_pos = self.get_position(3)
            if self.distance_between_points(cf2_pos, cf3_pos) < 1:
                    self.repel(cf2_pos, cf3_pos, cf_id=2, ob_id=3)
            
            # If multiple obstacles are within collison range, then avoid the closest
            # min_dist = np.min(dist_from_ob)
            # if min_dist < 1:
            #     min_index = np.argmin(dist_from_ob)
            #     print('Avoid: ', min_index + 2)

            # if self.avoidance_method == 'Heuristic':
            #     self.repel(cf_position, ob_position, i)
            # TO DO 
            # if self.avoidance_method == 'RL Separate':
            #     pass
            # if self.avoidance_method == 'RL Combined':
            #     pass

        # Restart simulation if drone has flipped
        if is_flipped:
            self.kill_sim()
            time.sleep(20)
            self.gazebo_process, self.cf_process = self.launch_sim()
        
        return observation, reward, is_terminal, {}
    

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

        # Reached goal
        if dist_to_goal < 1:
            reward += 50 
            is_terminal = True 
            print('REACHED GOAL.....')
        else:
            if self.steps == 256:
                is_terminal = True
            # Penalize based on distance to goal
            reward -= dist_to_goal
            if is_flipped:
                # Penalize if drone has flipped over
                # reward -= 200 
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
                vel_xy[i] = 1
            elif vel_xy[i] < 0:
                vel_xy[i] = -1
        
        # Get vertical avoidance
        z_increase = cf_position[2] >= ob_position[2]

        action_msg = Hover()
        action_msg.vx = vel_xy[0] 
        action_msg.vy = vel_xy[1] 
        action_msg.zDistance = cf_position[2] 
        action_msg.yawrate = 0 

        for _ in range(3):
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


    def get_position(self, cf_id):
        pose = None
        while pose is None:
            try:
                pose = rospy.wait_for_message('/cf' + str(cf_id) + '/local_position', GenericLogData, timeout=5)
            except:
                rospy.loginfo("Crazyflie pose not ready yet, retrying for getting robot pose")

        # get x,y,z position
        position = np.array(pose.values[:3])
        return position


    def get_velocities(self, cf_id):
        imu = None
        while imu is None:
            try:
                imu = rospy.wait_for_message('/cf' + str(cf_id) + '/imu', Imu, timeout=5)
            except:
                rospy.loginfo("Crazyflie 1 imu not ready yet, retrying for getting robot imu")

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

        launch_gazebo_cmd = 'roslaunch crazyflie_gazebo multiple_cf_sim_' + str(self.n_obstacles+1) + '.launch'
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
            target_position = cf_position + np.random.normal(-0.5, 0.5, size = 3)
            # Obstacle must wait 5 steps to move after it has been avoided
            if self.steps_since_avoided[i-2] == 0:
                self.cfs[i-1].goTo(goal=[target_position[0], target_position[1], np.clip(target_position[2], 0.5, 9.5)], yaw=0.0, duration=4)

        # Update steps since avoidance
        for i in range(self.n_obstacles):
            if self.steps_since_avoided[i] > 0:
                if self.steps_since_avoided[i] == 5:
                    self.steps_since_avoided[i] = 0
                else:
                    self.steps_since_avoided[i] += 1


    def process_action(self, action):
        """ Converts an array of actions into the necessary ROS msg type.

        Args:
            action (ndarray): Array containing the desired velocties along the 
            x, y, and z axes. 

        Returns:
            Hover: ROS msg type necessary to publish a velocity command.  
        """
        
        observation = self.get_observation()
        
        # Bounce drone off right wall
        if observation[0] >= 9.5 and action[0] > 0:
            action[0] = 0
            observation = self.get_observation()
        # Bounce drone off left wall
        elif observation[0] <= -9.5 and action[0] < 0:
            action[0] = 0
            observation = self.get_observation()

        # Bounce drone off back wall
        if observation[1] >= 9.5 and action[1] > 0:
            action[1] = 0
            observation = self.get_observation()
        # Bounce drone off front wall
        elif observation[1] <= -9.5 and action[1] < 0:
            action[1] = 0
            observation = self.get_observation()

        # Option 1: Hovering movements 
        action_msg = Hover()
        action_msg.vx = action[0] 
        action_msg.vy = action[1] 
        action_msg.zDistance = action[2]
        action_msg.yawrate = 0 

        # Option 2: Velocity movements 
        # action_msg = Twist()
        # action_msg.linear.x = action[0]
        # action_msg.linear.y = action[1]
        # action_msg.linear.z = action[2]
        # action_msg.angular.z = 0 

        # Option 3: Positon movements 
        # action_msg = FullState()
        # action_msg.pose.position.x = action[0] 
        # action_msg.pose.position.y = action[1]
        # action_msg.pose.position.z = action[2] 

        return action_msg 


    def distance_between_points(self, point_a, point_b):
        """ Returns the Euclidean distance between two points.

        Args:
            point_a (list): (x, y, z) coordinates of the first point. 
            point_a (list): (x, y, z) coordinates of the second point. 

        Returns:
            float: Euclidean distance between the two points.
        """
        return np.linalg.norm(point_a - point_b)
