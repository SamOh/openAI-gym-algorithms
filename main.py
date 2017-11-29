import gym
import os
import utils
from learningAgents import RandomAgent

### User Params ###

# The name of the game to solve
game_name = 'SpaceInvaders-v0'
number_iterations = 100
number_trials = 5
# 6 is for space invaders. Check the number potential for each game by printing action_space
number_potential_actions = 6

### End User Params ###

# hc_agent = HillClimbingAgent(game_name, number_iterations, number_potential_actions)
random_agent = RandomAgent(game_name, number_iterations, number_potential_actions)

# train agent here and save results of the best
print(random_agent.train_agent())
