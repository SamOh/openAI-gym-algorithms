import gym
from learningAgents import *

### TAXI PARAMS ###


game_name = 'Taxi-v2'
training_iterations = 2000
testing_iterations = 1000
epsilon = 0.15
gamma = 0.9
alpha = 0.1

ql_agent = QLearningAgent(game_name, training_iterations, testing_iterations, epsilon, gamma, alpha)
ql_agent.train_agent()
ql_agent.test_taxi()

ql_agent.resetQValues()
ql_agent.train_agent_taxi()
ql_agent.test_taxi()
