#!/usr/bin/env python3

import os
import cf_gym_env
import rospy
import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import CheckpointCallback

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        # env = Monitor(env, log_dir)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init


if __name__ == '__main__':
    rospy.init_node('drone_gym')
    env_id = 'Crazyflie-v0'
    num_cpu = 1 # Number of processes to use

    # Create the vectorized environment
    env = DummyVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    model = PPO2(MlpPolicy, env, verbose=1)

    # Load trained params and continue training. Use 'reset_num_timesteps=True' in model.learn()
    # model = PPO2.load('ppo2_crazyflie')
    # model.set_env(env)

    # Create the callback: check every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/hover/', name_prefix='ppo2')

    time_steps = 60000
    model.learn(total_timesteps=time_steps, callback=checkpoint_callback)