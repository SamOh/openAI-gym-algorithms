import gym
import os
import utils
from learningAgents import *

### User Params ###

# The name of the game to solve
# <<<<<<< HEAD
# game_name = 'SpaceInvaders-v0'
# number_iterations = 100
# number_trials = 1
# # 6 is for space invaders. Check the number potential for each game by printing action_space
# number_potential_actions = 6
# =======
game_name = 'MountainCarContinuous-v0'
number_iterations = 2000
epsilon = 0.08
gamma = 0.85
alpha = 0.1
# >>>>>>> 166aec55b03cfa14bbc8a4bcabe8c5321c61d090


### End User Params ###

# tdl_agent = TDLearningAgent(game_name, number_iterations, epsilon, gamma, alpha)
# tdl_agent.train_agent()
# tdl_agent.test_agent()

random_agent = RandomAgent(game_name)
random_agent.test_agent()

# # train agent here and save results of the best
# print(random_agent.train_agent())
