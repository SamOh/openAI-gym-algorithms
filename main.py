import gym
import os
from learningAgents import RandomAgent

### User Params ###

# The name of the game to solve
game_name = 'Taxi-v2'
number_iterations = 2000
epsilon = 0.05
gamma = 0.9
alpha = 0.5

tdl_agent = TDLearningAgent(game_name, number_iterations, epsilon, gamma, alpha)
tdl_agent.train_agent()
tdl_agent.test_agent()

# random_agent = RandomAgent(game_name)
# random_agent.test_agent()
