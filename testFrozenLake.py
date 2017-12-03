import gym
from learningAgents import *
from frozenLakeMDP import *
from valueIterationAgents import *

"""
FrozenLake 4x4 tests
"""

print 'Testing FrozenLake 4x4 Grid...\n'

print 'Testing Q-Learning...\n'

game_name = 'FrozenLake-v0'
training_iterations = 5000
testing_iterations = 1000
holes = {(2,2), (4,2), (4,3), (1,4)}

# Some hyperparameters tuned for frozen lake!
epsilon = 0.2
gamma = 0.99
alpha = 0.1

ql_agent = QLearningAgent(game_name, training_iterations, testing_iterations, \
                           epsilon, gamma, alpha)
ql_agent.train_agent()
ql_agent.test_frozen_lake(4)

ql_agent.resetQValues()
ql_agent.train_agent_frozen(4, holes)
ql_agent.test_frozen_lake(4)

print 'Testing Value Iteration...\n'

mdp = FrozenLakeMDP(4, holes)
valueAgent = ValueIterationAgent(mdp, testing_iterations)
valueAgent.test_agent(game_name, 4)

"""
FrozenLake 8x8 tests
"""

print 'Testing FrozenLake 8x8 Grid...\n'

print 'Testing Q Learning...\n'

game_name = 'FrozenLake8x8-v0'
holes = {(4,3),(6,4),(4,5),(2,6),(3,6),(7,6),(2,7),(5,7),(7,7),(4,8)}

ql_agent = QLearningAgent(game_name, training_iterations, testing_iterations, \
                           epsilon, gamma, alpha)
ql_agent.train_agent()
ql_agent.test_frozen_lake(8)

ql_agent.resetQValues()
ql_agent.train_agent_frozen(8, holes)
ql_agent.test_frozen_lake(8)

print 'Testing Value Iteration...\n'

mdp = FrozenLakeMDP(8, holes)
valueAgent = ValueIterationAgent(mdp, testing_iterations)
valueAgent.test_agent(game_name, 8)
