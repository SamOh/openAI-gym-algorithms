import gym
from learningAgents import *
from frozenLakeMDP import *
from valueIterationAgents import *

"""
FrozenLake 4x4 tests
"""

print 'Testing FrozenLake 4x4 Grid...'

print 'Testing Q-Learning...'

game_name = 'FrozenLake-v0'
number_iterations = 10000
epsilon = 0.2
gamma = 0.99
alpha = 0.1

tdl_agent = QLearningAgent(game_name, number_iterations, epsilon, gamma, alpha)
tdl_agent.train_agent()
tdl_agent.test_frozen_lake()

print 'Testing Value Iteration...'

mdp4x4 = FrozenLakeMDP(4, {(2,2), (4,2), (4,3), (1,4)})
discount = 0.9
iterations = 25
valueAgent = ValueIterationAgent(mdp4x4, discount, iterations)
test_episodes = 1000
valueAgent.test_agent(game_name, test_episodes)

# optimal policy list: note that None appears at holes.
policy = []
for state in range(16):
  policy.append(valueAgent.getPolicy(mdp4x4.getPos(state)))
print 'optimal policy: ', policy


"""
FrozenLake 8x8 tests
"""

print '\nTesting FrozenLake 8x8 Grid...'

print 'Testing Q Learning...'

game_name = 'FrozenLake8x8-v0'
number_iterations = 10000
epsilon = 0.2
gamma = 0.99
alpha = 0.1

tdl_agent = QLearningAgent(game_name, number_iterations, epsilon, gamma, alpha)
tdl_agent.train_agent()
tdl_agent.test_frozen_lake()

print 'Testing Value Iteration...'

mdp8x8 = FrozenLakeMDP(8, {(4,3),(6,4),(4,5),(2,6),(3,6),(7,6),(2,7),(5,7),(7,7),(4,8)})
discount = 0.9
iterations = 100
valueAgent = ValueIterationAgent(mdp8x8, discount, iterations)
test_episodes = 1000
valueAgent.test_agent(game_name, test_episodes)

# optimal policy list: note that None appears at holes.
policy = []
for state in range(64):
  policy.append(valueAgent.getPolicy(mdp8x8.getPos(state)))
print 'optimal policy: ', policy



