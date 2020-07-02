#!/usr/bin/env python3

import gym
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import PPO2
import cf_gym_env
import rospy


rospy.init_node('drone_gym', anonymous=True)

if __name__ == '__main__':
    env_id = 'Crazyflie-v0'
    env = gym.make(env_id)
    model = PPO2.load("ppo2_crazyflie")

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    # Observe trained agent 
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)