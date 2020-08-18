#!/usr/bin/env python3

import os
import cf_gym_env
import rospy
import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines import PPO2, results_plotter
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import BaseCallback, CheckpointCallback, StopTrainingOnRewardThreshold
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize



class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


def make_env(env_id, log_dir, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env = Monitor(env, log_dir)
        env.seed(seed)
        return env
    set_global_seeds(seed)
    return _init


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Learning Curve Smoothed")
    plt.show()

     
if __name__ == "__main__":
    rospy.init_node('drone_gym')
    env_id = 'Crazyflie-v0'
    log_dir = 'models/hover/empty_world_small/finalVec'

    env = DummyVecEnv([lambda: gym.make(env_id)])
    # Automatically normalize the input features and reward
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # # Save best model every n steps and monitors performance
    # save_best_callback = SaveOnBestTrainingRewardCallback(check_freq=5, log_dir=log_dir)
    # # Save model every n steps 
    # checkpoint_callback = CheckpointCallback(save_freq=5, save_path='./' + log_dir, name_prefix='ppo2')

    # Train from scratch
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=80000)
    # model.learn(total_timesteps=20, callback=[save_best_callback, checkpoint_callback])

    # Don't forget to save the VecNormalize statistics when saving the agent
    model.save(log_dir + "/ppo2_final")
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    env.save(stats_path)