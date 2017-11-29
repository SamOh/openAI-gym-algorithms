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

        # iterate through specified range and add up the total reward
        for _ in range(self.iterations):
            observation, reward, done, info = self.env.step(self.env.action_space.sample())
            total_reward += reward
            if done:
                break

        return total_reward

    def train_agent(self):
        best_reward = 0

        for _ in range(self.iterations):
            reward = self.random_action(self.env)

            if reward > best_reward:
                best_reward = reward

        return best_reward



"""
Basic Greedy hill climbing agent
"""
class HillClimbingAgent(Agent):
    def __init__(self, game_name, iterations, num_actions):
        self.env = gym.make(game_name)
        self.iterations = iterations
        self.totalreward = 0
        self.observation = self.env.reset()
        self.num_actions = num_actions

    # choose the best action and calculate the step function
    def best_step(self, env):
        best_reward, best_obs, best_done, best_info = 0, None, None, None

        for action in range(self.num_actions):
            copy_env = copy.deepcopy(env)
            observation, reward, done, info = copy_env.step(action)
            if reward > best_reward:
                best_reward = reward
                best_obs = observation
                best_done = done
                best_info = info

        return best_obs, best_reward, best_done, best_info

    def hill_climbing(self, env):
        observation = env.reset()
        total_reward = 0

        # iterate through specified range and add up the total reward
        for _ in range(self.iterations):
            observation, reward, done, info = self.best_step(env)
            total_reward += reward
            if done:
                break

        return total_reward

    def train_agent(self):
        best_reward = 0

        for _ in range(self.iterations):
            reward = self.hill_climbing(self.env)

            if reward > best_reward:
                best_reward = reward

        return best_reward


"""
Basic QLearningAgent for games
"""
class SimulatedAnnealingAgent(Agent):
    def __init__(self, game_name, iterations):
        # instantiate Q values
        pass

    # probability based on T and badness of move
    def probability_function(ratio_difference, T):
        return T + ratio_difference

    # temperature function decreases with time
    def temperature_function(time):
        return 1.0 / time

    def simulted_annealing():
        # TODO
        pass

    def train_agent(env):
        # do the training of the agent here
        pass

