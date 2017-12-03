import gym
import random
import utils
from taxiSearch import *

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
Agent for Q-Learning
"""
class QLearningAgent(LearningAgent):
    def __init__(self, game_name, training_iterations, testing_iterations, epsilon, gamma, alpha):
        self.env = gym.make(game_name)
        self.training = training_iterations
        self.testing = testing_iterations
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.QValues = utils.Counter()
        self.actions = [i for i in range(self.env.action_space.n)]

    def resetQValues(self):
        self.QValues = utils.Counter()

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
        print 'Starting regular QLearning with {} iterations...'.format(self.training)
        for episode in range(self.training):
            done, prevObs = False, self.env.reset()
            while not done:
                action = self.epsilonGreedyAction(prevObs)
                obs, reward, done, _ = self.env.step(action)
                self.updateQValues(prevObs, action, obs, reward)
                prevObs = obs

    def test_frozen_lake(self, maxValue):
        rewards = 0
        for _ in range(self.testing):
            obs, done = self.env.reset(), False
            while not done:
                action = self.getPolicy(obs)
                obs, reward, done, info = self.env.step(action)
                rewards += reward
        print "Percent success rate was {}%\n".format(rewards*100.0/self.testing)
        policy = []
        for i in range(maxValue*maxValue):
            policy.append(self.getPolicy(i))
        print "Learned Policy: {}\n".format(policy)

    def test_taxi(self):
        passed = 0
        for _ in range(self.testing):
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
        print 'Optimal performance rate was {}%\n'.format(passed*100.0/self.testing)

"""
QLearning with extra incentives for taxi
"""
class TaxiQLAgent(QLearningAgent):
    def __init__(self, game_name, training_iterations, testing_iterations, epsilon, gamma, alpha):
        QLearningAgent.__init__(self, game_name, training_iterations, testing_iterations, epsilon, gamma, alpha)

    def train_agent(self):
        print 'Starting special QLearning for Taxi with {} iterations...'.format(self.training)
        for episode in range(self.training):
            done, prevObs = False, self.env.reset()
            pickup, dropoff = findPickupDropoff(prevObs)

            while not done:
                action = self.epsilonGreedyAction(prevObs)
                obs, reward, done, _ = self.env.step(action)

                pos, prevPos = getPosition(obs), getPosition(prevObs)
                if isPickingUp(obs):
                    dist, prevDist = manhattanDistance(pos, pickup), manhattanDistance(prevPos, pickup)
                    if dist < prevDist:
                        reward += 1
                    if dist > prevDist:
                        reward -= 1
                else:
                    dist, prevDist = manhattanDistance(pos, dropoff), manhattanDistance(prevPos, dropoff)
                    if dist < prevDist:
                        reward += 1
                    if dist > prevDist:
                        reward -= 1

                self.updateQValues(prevObs, action, obs, reward)
                prevObs = obs

"""
QLearning with extra incentives for FrozenLake
"""
class FrozenLakeQLAgent(QLearningAgent):
    def __init__(self, game_name, training_iterations, testing_iterations, epsilon, gamma, alpha):
        QLearningAgent.__init__(self, game_name, training_iterations, testing_iterations, epsilon, gamma, alpha)

    def train_agent(self, maxValue, holes):
        print 'Starting special QLearning for FrozenLake with {} iterations...'.format(self.training)

        for episode in range(self.training):
            done, prevObs = False, self.env.reset()
            while not done:
                action = self.epsilonGreedyAction(prevObs)
                obs, reward, done, _ = self.env.step(action)

                # reward getting closer to the goal
                if ((obs % maxValue) > (prevObs % maxValue) or \
                (obs / maxValue) > (prevObs / maxValue)):
                    reward += 0.01

                # punish going backwards or dying
                if ((obs % maxValue) < (prevObs % maxValue) or \
                (obs / maxValue) < (prevObs / maxValue)) or obs in holes:
                    reward -= 0.01

                self.updateQValues(prevObs, action, obs, reward)
                prevObs = obs

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

