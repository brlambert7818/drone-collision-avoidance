#!/usr/bin/env python3

import os
import cf_obstacle_gym_env
import rospy
import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines import PPO2, results_plotter
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import BaseCallback, CheckpointCallback
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


def make_env(env_id, rank, log_dir, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, n_obstacles=2, avoidance_method='Heuristic')
        env = Monitor(env, log_dir)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


if __name__ == '__main__':

    rospy.init_node('drone_gym')
    env_id = 'CrazyflieObstacle-v0'
    log_dir = 'models/hover/empty_world_small/heuristic'
    num_cpu = 1  # Number of processes to use

    # Create the vectorized environment
    env = DummyVecEnv([make_env(env_id, i, log_dir) for i in range(num_cpu)])
    env = VecNormalize(env)

    # Save best model every n steps and monitors performance
    save_best_callback = SaveOnBestTrainingRewardCallback(check_freq=500, log_dir=log_dir)
    # Save model every n steps 
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./' + log_dir, name_prefix='ppo2')

    # Train from scratch
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=200000, callback=save_best_callback)

    # Load trained params and continue training
    # model = PPO2.load(log_dir + '/best_model')
    # model.set_env(env)
    # model.learn(total_timesteps=60000, callback=save_best_callback)

    results_plotter.plot_results([log_dir], 200000, results_plotter.X_TIMESTEPS, "PPO Crazyflie")
    plt.show()
     
    env.close()