import gym
import random
import utils
import numpy as np
from taxi import solveTaxi

"""
General class to inherit for other learning algo classes
"""
class LearningAgent:
    def __init__(self):
        pass

    def train_agent(self):
        """
        All agents must have a training component that trains the agent to learn how to play
        the game based on which algorithm is being used
        """
        pass

    def test_agent(self):
        pass

"""
Random Agent to compare with (takes random actions)
"""
class RandomAgent(object):
    def __init__(self, game_name):
        self.env = gym.make(game_name)

    def test_agent(self):
        print('testing RandomAgent...')
        self.env.reset()
        done, episode_rewards = False, 0
        while not done:
            _, reward, done, _ = self.env.step(self.env.action_space.sample())
            episode_rewards += reward
        print 'testing episode gained {} rewards'.format(episode_rewards)


"""
Agent for Temporal Difference Learning
"""
class TDLearningAgent(LearningAgent):
    def __init__(self, game_name, iterations, epsilon, gamma, alpha):
        self.env = gym.make(game_name)
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.iterations = iterations
        self.QValues = utils.Counter()
        self.actions = [i for i in range(self.env.action_space.n)]

    def getQValue(self, state, action):
        return self.QValues[state, action]

    def getValue(self, state):
        maxQValue = None
        for action in self.actions:
          QValue = self.getQValue(state, action)
          if maxQValue < QValue:
            maxQValue = QValue
        return 0 if maxQValue is None else maxQValue

    def getPolicy(self, state):
        maxQValue, maxAction = None, None
        for action in self.actions:
          QValue = self.getQValue(state, action)
          if maxQValue < QValue:
            maxQValue = QValue
            maxAction = action
        return maxAction

    def epsilonGreedyAction(self, state):
        return random.choice(self.actions) if utils.flipCoin(self.epsilon) else self.getPolicy(state)

    def updateQValues(self, state, action, nextState, reward):
        self.QValues[state, action] = (1 - self.alpha) * self.QValues[state, action] + \
          self.alpha * (reward + self.gamma * self.getValue(nextState))

    def train_agent(self):
        print 'training TDLearningAgent with {} iterations...'.format(self.iterations)
        for episode in range(self.iterations):
            episode_rewards = 0
            done, prevObs = False, self.env.reset()
            while not done:
                action = self.epsilonGreedyAction(prevObs)
                obs, reward, done, _ = self.env.step(action)
                self.updateQValues(prevObs, action, obs, reward)
                prevObs = obs
                episode_rewards += reward
            #print 'training episode gained {} rewards in episode {}'.format(episode_rewards, episode)

    def test_agent(self):
        print 'testing TDLearningAgent...'
        passed = 0
        for _ in range(self.iterations):
            obs, done, actions = self.env.reset(), False, []
            optimalActions = solveTaxi(obs)
            while not done:
                action = self.getPolicy(obs)
                actions.append(action)
                obs, _, done, _ = self.env.step(action)
                if len(actions) > len(optimalActions):
                    break
            if done:
                if len(actions) == len(optimalActions):
                    passed += 1
                if len(actions) < len(optimalActions):
                    print "This is impossible!"
        print '{}% of tests passed optimally'.format(passed * 100.0 / self.iterations)


"""
TODO
"""
class MonteCarloAgent(LearningAgent):
    def __init__(self, game_name, iterations, epsilon, gamma, alpha):
        self.env = gym.make(game_name)
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.iterations = iterations
        self.QValues = utils.Counter()
        self.actions = [i for i in range(self.env.action_space.n)]

    def train_agent(self):
        return

class DiscreteInputBasicQLearning(Agent):
    def __init__(self, game_name, iterations, epsilon, gamma, alpha):
        self.env = gym.make(game_name)
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.iterations = iterations
        self.QValues = np.zeros([env.observation_space.n,env.action_space.n])
        self.actions = [i for i in range(self.env.action_space.n)]

    def getQValue(self, state, action):
        return self.QValues[state, action]

    def getValue(self, state):
        maxQValue = None
        for action in self.actions:
          QValue = self.getQValue(state, action)
          if maxQValue < QValue:
            maxQValue = QValue
        return 0 if maxQValue is None else maxQValue

    def getPolicy(self, state):
        maxQValue, maxAction = None, None
        for action in self.actions:
          QValue = self.getQValue(state, action)
          if maxQValue < QValue:
            maxQValue = QValue
            maxAction = action
        return maxAction

    def epsilonGreedyAction(self, state):
        return random.choice(self.actions) if utils.flipCoin(self.epsilon) else self.getPolicy(state)

    def updateQValues(self, state, action, nextState, reward):
        self.QValues[state, action] = (1 - self.alpha) * self.QValues[state, action] + \
          self.alpha * (reward + self.gamma * self.getValue(nextState))

    def train_agent(self):
        print 'training TDLearningAgent with {} iterations...'.format(self.iterations)
        for episode in range(self.iterations):
            episode_rewards = 0
            done, prevObs = False, self.env.reset()
            while not done:
                action = self.epsilonGreedyAction(prevObs)
                obs, reward, done, _ = self.env.step(action)
                self.updateQValues(prevObs, action, obs, reward)
                prevObs = obs
                episode_rewards += reward
            #print 'training episode gained {} rewards in episode {}'.format(episode_rewards, episode)

    def test_agent(self):
        print 'testing TDLearningAgent...'
        passed = 0
        for _ in range(self.iterations):
            obs, done, actions = self.env.reset(), False, []
            optimalActions = solveTaxi(obs)
            while not done:
                action = self.getPolicy(obs)
                actions.append(action)
                obs, _, done, _ = self.env.step(action)
                if len(actions) > len(optimalActions):
                    break
            if done:
                if len(actions) == len(optimalActions):
                    passed += 1
                if len(actions) < len(optimalActions):
                    print "This is impossible!"
        print '{}% of tests passed optimally'.format(passed * 100.0 / self.iterations)

    # def run(self):
    #     for episode in range(self.episodes):
    #         self.observation = self.env.reset()
    #         done = False
    #         total_reward, reward = 0,0

    #         action = numpy.argmax(self.Q[self.observation])

    #         observation, reward, done, info = self.env.step(self.env.action_space.sample())
    #         Q[self.observation, action] += alpha * (reward + gamma * (numpy.max(Q[observation] - Q[self.observation, action])))
    #         self.observation = observation
    #         total_reward += reward

    #     print "Score over time: " +  str(sum(rList)/num_episodes)

