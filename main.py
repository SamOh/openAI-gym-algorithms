import gym
import os
import utils
from learningAgents import *

### User Params ###

# The name of the game to solve
game_name = 'Taxi-v2'
number_iterations = 1000
epsilon = 0.1
gamma = 0.8
alpha = 0.2

### End User Params ###

tdl_agent = TDLearningAgent(game_name, number_iterations, epsilon, gamma, alpha)
tdl_agent.train_agent()
tdl_agent.test_agent()


# # hc_agent = HillClimbingAgent(game_name, number_iterations, number_potential_actions)
# random_agent = RandomAgent(game_name, number_iterations, number_potential_actions)

# # train agent here and save results of the best
# print(random_agent.train_agent())
