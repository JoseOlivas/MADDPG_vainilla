import numpy as np
from make_env import make_env

env = make_env('simple_adversary')

print('number of agents', env.n)
print('observation space', env.observation_space[0].shape)
print('action space', env.action_space)
print('n actions', env.action_space[0].n)

observation = env.reset()

print(observation)