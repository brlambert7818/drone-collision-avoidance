from gym.envs.registration import register

register(
    id='Crazyflie-v0',
    entry_point='gym_cf.envs:CrazyflieEnv',
)