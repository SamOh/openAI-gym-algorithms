import gym
import random
import cv2

"""
Maximize your score in the Atari 2600 game SpaceInvaders. In this environment,
the observation is an RGB image of the screen, which is an array of shape (210, 160, 3).
Each action is repeatedly performed for a duration of k frames, where k is
uniformly sampled from {2,3,4}.
"""

# 0 -> left
# 1 -> down
# 2 -> right
# 3 -> up

env = gym.make('FrozenLake-v0')
obs = env.reset()

env.render()
print obs
rewards, steps = 0, 0


while True:

  obs, r, d, info = env.step(3)
  env.render()
  print obs

  if d:
    break
  steps += 1
  rewards += r

print 'game ended with {} rewards in {} steps'.format(rewards, steps)

