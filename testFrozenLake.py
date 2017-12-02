import gym
from learningAgents import *
from frozenLakeMDP import *
from valueIterationAgents import *

"""
QLearning Tests
"""

game_name = 'FrozenLake-v0'
number_iterations = 10000
epsilon = 0.2
gamma = 0.99
alpha = 0.1

tdl_agent = TDLearningAgent(game_name, number_iterations, epsilon, gamma, alpha)
tdl_agent.train_agent()
tdl_agent.test_frozen_lake()

"""
Value Iteration Tests
"""

print 'Testing Value Iteration...'

mdp = FrozenLakeMDP()
discount = 0.9
iterations = 25
valueAgent = ValueIterationAgent(mdp, discount, iterations)
test_episodes = 1000
valueAgent.test_agent(game_name, test_episodes)



