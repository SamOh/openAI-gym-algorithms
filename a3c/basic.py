import gym
import random
import cv2

env = gym.make('Pong-v0')
env.reset()
env.render()
steps = 0
rewards = 0
currentLives = 3

d = False
while not d:
  s, r, d, info = env.step(env.action_space.sample())
  env.render()

print('game ended with {} rewards in {} steps'.format(rewards, steps))

