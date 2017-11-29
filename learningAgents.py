import gym
import os
import random
import numpy as np
import copy

"""
General class to inherit for other learning algo classes
"""
class Agent:
    def __init__(self):
        pass

    def train_agent(self, env):
        """
        All agents must have a training component that trains the agent to learn how to play
        the game based on which algorithm is being used
        """
        pass

"""
Random Agent to compare with (takes random actions)
"""
class RandomAgent(Agent):
    def __init__(self, game_name, iterations, num_actions):
        self.env = gym.make(game_name)
        self.iterations = iterations
        self.totalreward = 0
        self.observation = self.env.reset()
        self.num_actions = num_actions

    def random_action(self, env):
        observation = env.reset()
        total_reward = 0
        # print(observation)

        # iterate through specified range and add up the total reward
        for _ in range(self.iterations):
            action = self.env.action_space.sample()
            # print(action)
            observation, reward, done, info = self.env.step(action)
            # print(info)
            # print(reward)
            total_reward += reward
            if done:
                break

        return total_reward

    def train_agent(self):
        best_reward = 0

        for _ in range(self.iterations):
            self.env.render()
            reward = self.random_action(self.env)
            if reward > best_reward:
                best_reward = reward

        return best_reward


"""
Basic Estimated QLearning (RL2 last thing scott talked about)
"""
class ApproxamiteQLearningAgent(Agent):
    def __init__(self, game_name, iterations):
        # instantiate Q values
        pass

    def train_agent(env):
        # do the training of the agent here
        pass

