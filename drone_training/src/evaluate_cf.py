#!/usr/bin/env python3

import os
import gym
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import PPO2
import cf_obstacles_gym_env_eval
import cf_gym_env
import rospy
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize


rospy.init_node('drone_gym', anonymous=True)


def make_env(env_id, rank, log_dir, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, n_obstacles=1, avoidance_method='Heuristic')
        env = Monitor(env, log_dir)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


if __name__ == '__main__':
    log_dir = 'models/hover/empty_world_small/finalVec'
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    env_id = 'CrazyflieObstacleEval-v0'

    # Load the agent
    model = PPO2.load(log_dir + '/ppo2_final')

    # Load the saved statistics
    env = DummyVecEnv([lambda: gym.make(env_id, n_obstacles=1, avoidance_method='Heuristic')])
    env = VecNormalize.load(stats_path, env)
    #  do not update them at test time
    env.training = False
    # reward normalization is not needed at test time
    env.norm_reward = False

    eval_episodes = 50 

    total_goals_reached = 0
    total_collisions = 0
    total_flips = 0
    total_steps_exceeded = 0
    total_potential_collisions = 0
    total_collisions_avoided = 0
    total_timsteps = 0

    # Observe trained agent 
    for i_episode in range(eval_episodes):
        is_terminal = False
        n_steps = 0

        obs = env.reset()
        while not is_terminal:  
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            n_steps += 1

            if info[0]["reached_goal"]:
                total_goals_reached += 1    

            if info[0]["needs_to_avoid"]:
                if info[0]['avoided_collision']:
                    total_potential_collisions += 1 
                    total_collisions_avoided += 1
                else:
                    total_potential_collisions += 1 
                    total_collisions += 1
            
            if info[0]['flipped']:
                total_flips += 1
            
            if info[0]['exceeded_steps']:
                total_steps_exceeded += 1
            
            is_terminal = dones
        
        total_timsteps += n_steps
    
    percent_goals_reached = total_goals_reached / eval_episodes 
    percent_collision = total_collisions / eval_episodes 
    percent_flipped = total_flips / eval_episodes 
    percent_steps_exceeded = total_steps_exceeded / eval_episodes 
    percent_collisions_avoided = total_collisions_avoided /total_potential_collisions
    avg_episode_length = total_timsteps / eval_episodes

    eval_dict = {
        "percent_goals_reached": percent_goals_reached,
        "percent_collision": percent_collision,
        "percent_flipped": percent_flipped,
        "percent_steps_exceeded": percent_steps_exceeded,
        "percent_collisions_avoided": percent_collisions_avoided,
        "avg_episode_length": avg_episode_length
    }

    for k,v in eval_dict.items():
        print(k + ": ", v)
