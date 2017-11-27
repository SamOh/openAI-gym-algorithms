import gym
import os
import learningAgents

### User Params ###

# The name of the game to solve
game_name = 'SpaceInvaders-ram-v0'

### End User Params ###


# Instantiate the different classes in learning Agents here and run the training at the bottom after making env




env = gym.make(game_name)

# train agent here and save results of the best
train_agent(env)
