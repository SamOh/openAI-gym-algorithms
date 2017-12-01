import gym
import utils
from learningAgents import * 

### TAXI PARAMS ###

game_name = 'FrozenLake-v0'
number_iterations = 1
epsilon = 0.15
gamma = 0.9
alpha = 0.1




tdl_agent = TDLearningAgent(game_name, number_iterations, epsilon, gamma, alpha)
tdl_agent.train_agent()
tdl_agent.test_frozen_lake()

# random_agent = RandomAgent(game_name)
# random_agent.test_agent()

