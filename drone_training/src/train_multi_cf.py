#!/usr/bin/env python3

import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2
import cf_multi_gym_env
import rospy

from gazebo_connection import GazeboConnection
import subprocess
import time
import os


def make_env(env_id, rank, gazebo, gazebo_process, cf_process, seed=1):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, cf_id=rank+1, gazebo=gazebo, gazebo_process=gazebo_process, cf_process=cf_process)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init


if __name__ == '__main__':

    rospy.init_node('drone_gym', anonymous=True)

    # Launch Gazebo process 
    launch_gazebo_cmd = 'roslaunch crazyflie_gazebo multiple_cf_sim.launch'
    gazebo_process = subprocess.Popen(launch_gazebo_cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
    time.sleep(5)

    # Launch Crazyflie controllers
    cf_gazebo_path = '/home/brian/catkin_ws/src/sim_cf/crazyflie_gazebo/scripts'
    launch_controller_cmd = './run_cfs.sh 4'
    cf_process = subprocess.Popen(launch_controller_cmd, stdout=subprocess.PIPE, cwd=cf_gazebo_path, shell=True, preexec_fn=os.setsid)
    time.sleep(5)

    gazebo = GazeboConnection()

    env_id = 'Crazyflie-v1'
    num_cpu = 4 # Number of processes to use
    # Create the vectorized environment
    env = DummyVecEnv([make_env(env_id, i, gazebo, os.getpgid(gazebo_process.pid), os.getpgid(cf_process.pid)) for i in range(num_cpu)])

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=30000)
    model.save('ppo2_crazyflie_multi')