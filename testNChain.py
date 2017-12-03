import gym
from learningAgents import *
from frozenLakeMDP import *
from valueIterationAgents import *
from NChainMDP import *

"""
NChain tests
"""

print 'Testing NChain...\n'

# print 'Testing Q-Learning...\n'

game_name = 'NChain-v0'
training_iterations = 5000
testing_iterations = 1000

# hyperparameters for NChain
epsilon = 1
gamma = 0.99
alpha = 0.1

ql_agent = QLearningAgent(game_name, training_iterations, testing_iterations, \
                           epsilon, gamma, alpha)
ql_agent.train_agent()
ql_agent.test_nchain()

print 'Testing Random Solution...\n'

random_agent = RandomAgent(game_name)
random_agent.test_agent()

print 'Testing Backwards Policy Solution...\n'

env = gym.make(game_name)
total_reward = 0
for _ in range(testing_iterations):
    obs, done, actions = env.reset(), False, []
    obs, reward, done, info = env.step(1)
    total_reward += reward

print total_reward

print 'Testing Forwards Policy Solution...\n'

env = gym.make(game_name)
total_reward = 0
for _ in range(testing_iterations):
    obs, done, actions = env.reset(), False, []
    obs, reward, done, info = env.step(0)
    total_reward += reward

print total_reward






