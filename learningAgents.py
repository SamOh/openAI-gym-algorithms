import gym
import random
import utils
from taxiSearch import solveTaxi
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
        print 'testing RandomAgent...'
        self.env.reset()
        done, episode_rewards = False, 0

        while not done:
            _, reward, done, _ = self.env.step(self.env.action_space.sample())
            episode_rewards += reward
        print 'testing episode gained {} rewards'.format(episode_rewards)


"""
Agent for Temporal Difference Learning
"""
class QLearningAgent(LearningAgent):
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
            # print 'training episode gained {} rewards in episode {}'.format(episode_rewards, episode)

    def test_frozen_lake(self):
        rewards, iterations = 0, 1000
        for _ in range(iterations):
            obs, done = self.env.reset(), False
            while not done:
                action = self.getPolicy(obs)
                obs, reward, done, info = self.env.step(action)
                rewards += reward
        print "Percent success rate was {}%".format(rewards*100.0/iterations)

    def test_taxi(self):
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
        print 'Percent success rate was {}%'.format(passed*100.0/self.iterations)

"""
TODO
"""


class MonteCarloAgent(LearningAgent):
    def __init__(self, game_name, iterations, gamma):
        self.env = gym.make(game_name)
        self.gamma = gamma
        self.iterations = iterations
        self.state_rewards = {}
        self.state_value = utils.Counter()
        self.actions = [i for i in range(self.env.action_space.n)]

    def train_agent(self):
        print 'training MCLearningAgent with {} iterations...'.format(self.iterations)
        for episode in range(self.iterations):
            return

    def initializeRandomPolicy(self):
        self.env.reset()

        action, done = self.env.action_space.sample(), False

        observation, reward, done, info = self.env.step(self.env.action_space.sample())
        while done == False:
            if observation in self.state_value:
                self.state_rewards[observation].append(reward)
            else:
                self.state_rewards[observation] = [reward]
            observation, reward, done, info = self.env.step(self.env.action_space.sample())

        self.state_value = utils.Counter()
        for obs in self.state_rewards.keys():
            self.state_value = sum(self.state_rewards[obs]) / float( len(self.state_rewards[obs]) )

        self.state_rewards = {} # ERASE


    def updatePolicy(self):
        return

    def getPolicy(self, state):
        self.env.action_space.sample()
        self.state_value[state]



        maxQValue, maxAction = None, None
        for action in self.actions:
          QValue = self.getQValue(state, action)
          if maxQValue < QValue:
            maxQValue = QValue
            maxAction = action
        return maxAction





"""
Basic Estimated QLearning (RL2 last thing scott talked about)
"""
class ApproximateQLearningAgent(LearningAgent):
    def __init__(self, game_name, iterations):
        # instantiate Q values
        pass

    def train_agent(env):
        # do the training of the agent here
        pass
