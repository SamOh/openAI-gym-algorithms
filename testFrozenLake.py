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

tdl_agent = QLearningAgent(game_name, number_iterations, epsilon, gamma, alpha)
tdl_agent.train_agent()
tdl_agent.test_frozen_lake()

"""
Value Iteration Tests
"""

print 'Testing Value Iteration...'

mdp = FrozenLakeMDP(4, {(2,2), (4,2), (4,3), (1,4)})
discount = 0.9
iterations = 25
valueAgent = ValueIterationAgent(mdp, discount, iterations)
test_episodes = 1000
valueAgent.test_agent(game_name, test_episodes)

# optimal policy list: note that None appears at holes.
policy = []
for state in range(16):
  policy.append(valueAgent.getPolicy(mdp.getPos(state)))
print 'optimal policy: ', policy

print '\nTesting VI for 8x8 FrozenLake...'

game_name = 'FrozenLake8x8-v0'
mdp = FrozenLakeMDP(8, {(4,3),(6,4),(4,5),(2,6),(3,6),(7,6),(2,7),(5,7),(7,7),(4,8)})
discount = 0.99
iterations = 100
valueAgent = ValueIterationAgent(mdp, discount, iterations)
test_episodes = 100
valueAgent.test_agent(game_name, test_episodes)

# optimal policy list: note that None appears at holes.
policy = []
for state in range(16):
  policy.append(valueAgent.getPolicy(mdp.getPos(state)))
print 'optimal policy: ', policy



