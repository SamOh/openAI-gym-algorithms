import gym
import random

"""
Maximize your score in the Atari 2600 game SpaceInvaders. In this environment,
the observation is an RGB image of the screen, which is an array of shape (210, 160, 3).
Each action is repeatedly performed for a duration of k frames, where k is
uniformly sampled from {2,3,4}.
"""

env = gym.make('SpaceInvaders-v0')
state = env.reset()
steps = 0
rewards = 0

while True:
  state, reward, done, info = env.step(env.action_space.sample())
  if done:
    print info
    break
  steps += 1
  rewards += reward

print 'game ended with {} rewards in {} steps'.format(total_rewards, steps)
