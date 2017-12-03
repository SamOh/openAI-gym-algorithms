import gym
from learningAgents import *

### TAXI PARAMS ###


game_name = 'Taxi-v2'
number_iterations = 1250
epsilon = 0.15
gamma = 0.9
alpha = 0.1

tdl_agent = QLearningAgent(game_name, number_iterations, epsilon, gamma, alpha)
tdl_agent.train_agent()
tdl_agent.test_taxi()
