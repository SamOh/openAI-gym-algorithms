import gym
import os
import utils
from learningAgents import *
import tensorflow as tf
from learningAgents import RandomAgent
import numpy as np
import random
import tensorflow.contrib.slim as slim
import scipy.misc
import os
import itertools

### User Params ###

# The name of the game to solve
game_name = 'Taxi-v2'
number_iterations = 2000
epsilon = 0.08
gamma = 0.85
alpha = 0.1

### End User Params ###

tdl_agent = TDLearningAgent(game_name, number_iterations, epsilon, gamma, alpha)
tdl_agent.train_agent()
tdl_agent.test_agent()

# random_agent = RandomAgent(game_name)
# random_agent.test_agent()

# # train agent here and save results of the best
# print(random_agent.train_agent())
