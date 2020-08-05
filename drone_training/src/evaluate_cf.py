#!/usr/bin/env python3

import gym
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import PPO2
import cf_obstacles_gym_env
import rospy


rospy.init_node('drone_gym', anonymous=True)

if __name__ == '__main__':
    env_id = 'CrazyflieObstacle-v0'
    env = gym.make(env_id, n_obstacles=1, avoidance_method='None')
    log_dir = 'models/hover/empty_world_small/rl_obstacles'
    model = PPO2.load(log_dir + '/best_model')

    # Evaluate the agent
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    # Observe trained agent 
    obs = env.reset()
    for _ in range(500):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)