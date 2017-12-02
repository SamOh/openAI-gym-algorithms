import gym
from learningAgents import *
from frozenLakeMDP import *
from valueIterationAgents import *

"""
QLearning Tests
"""

game_name = 'FrozenLake-v0'
number_iterations = 5000
epsilon = 0.05
gamma = 0.9
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
iterations = 6
valueAgent = ValueIterationAgent(mdp, discount, iterations)

def getPos(i):
    return ((i % 4) + 1, (i / 4) + 1)

env = gym.make('FrozenLake-v0')
rewards, episodes = 0, 1000
for _ in range(episodes):
  obs, done = env.reset(), False
  while not done:
    action = valueAgent.getPolicy(getPos(obs))
    obs, reward, done, _ = env.step(action)
    rewards += reward
print 'Percent success rate was {}%'.format(rewards*100.0/episodes)



