#!/usr/bin/env python3

import os
import subprocess
import time
import rospy
import gym
import numpy as np
import matplotlib.pyplot as plt

import cf_multi_gym_env
from gazebo_connection import GazeboConnection

from stable_baselines import PPO2, results_plotter
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv, VecEnv


# class CustomCallback(BaseCallback):
#     """
#     A custom callback that derives from ``BaseCallback``.

#     :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
#     """
#     def __init__(self, verbose=0):
#         super(CustomCallback, self).__init__(verbose)
#         # Those variables will be accessible in the callback
#         # (they are defined in the base class)
#         # The RL model
#         # self.model = None  # type: BaseRLModel
#         # An alias for self.model.get_env(), the environment used for training
#         # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
#         # Number of time the callback was called
#         # self.n_calls = 0  # type: int
#         # self.num_timesteps = 0  # type: int
#         # local and global variables
#         # self.locals = None  # type: Dict[str, Any]
#         # self.globals = None  # type: Dict[str, Any]
#         # The logger object, used to report things in the terminal
#         # self.logger = None  # type: logger.Logger
#         # # Sometimes, for event callback, it is useful
#         # # to have access to the parent object
#         # self.parent = None  # type: Optional[BaseCallback]

#     def _on_step(self) -> bool:
#         """
#         This method will be called by the model after each call to `env.step()`.

#         For child callback (of an `EventCallback`), this will be called
#         when the event is triggered.

#         :return: (bool) If the callback returns False, training is aborted early.
#         """
#         return True


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


def make_env(env_id, rank, n_obstacles, log_dir, gazebo, gazebo_process, cf_process, seed=1):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, cf_id=rank+1, n_obstacles=n_obstacles, gazebo=gazebo, gazebo_process=gazebo_process, cf_process=cf_process)
        env = Monitor(env, log_dir)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


if __name__ == '__main__':

    rospy.init_node('drone_gym', anonymous=True)

    env_id = 'CrazyflieMulti-v0'
    log_dir = 'models/hover/empty_world_small/vectorized'
    num_cpu = 4 # Number of processes to use

    # Launch Gazebo process 
    launch_gazebo_cmd = 'roslaunch crazyflie_gazebo multiple_cf_sim_' + str(num_cpu) + '.launch'
    gazebo_process = subprocess.Popen(launch_gazebo_cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
    time.sleep(5)

    # Launch Crazyflie controllers
    cf_gazebo_path = '/home/brian/catkin_ws/src/sim_cf/crazyflie_gazebo/scripts'
    launch_controller_cmd = './run_cfs.sh ' + str(num_cpu)
    cf_process = subprocess.Popen(launch_controller_cmd, stdout=subprocess.PIPE, cwd=cf_gazebo_path, shell=True, preexec_fn=os.setsid)
    time.sleep(5)

    gazebo = GazeboConnection()
    gazebo.unpauseSim()

    # Create the vectorized environment
    env = DummyVecEnv([make_env(env_id, i, num_cpu-1, log_dir, gazebo, os.getpgid(gazebo_process.pid), os.getpgid(cf_process.pid)) for i in range(num_cpu)])
    env = VecNormalize(env)

    # Save best model every n steps and monitors performance
    # save_best_callback = SaveOnBestTrainingRewardCallback(check_freq=250, log_dir=log_dir)
    # Save model every n steps 
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./' + log_dir, name_prefix='ppo2')

    # Train from scratch
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=200000, callback=checkpoint_callback)

    # Load trained params and continue training
    # model = PPO2.load(log_dir + '/best_model')
    # model.set_env(env)
    # model.learn(total_timesteps=200000, callback=save_best_callback, reset_num_timesteps=False)

    results_plotter.plot_results([log_dir], 200000, results_plotter.X_TIMESTEPS, "PPO Crazyflie")
    plt.show()
     
    env.close()