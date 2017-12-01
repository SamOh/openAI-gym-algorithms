import gym
import os
import random
import numpy
import tensorflow
import copy
import utils

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
    def __init__(self, game_name):
        self.env = gym.make(game_name)

    def train_agent(self):
        return

    def test_agent(self):
        print 'testing RandomAgent...'
        self.env.reset()
        done, episode_rewards = False, 0
        while done == False:
            _, reward, done, _ = self.env.step(self.env.action_space.sample())
        print 'testing episode gained {} rewards'.format(episode_rewards)


"""
Agent for Temportal Difference Learning
"""
class TDLearningAgent(Agent):
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
            while done == False:
                action = self.epsilonGreedyAction(prevObs)
                obs, reward, done, _ = self.env.step(action)
                self.updateQValues(prevObs, action, obs, reward)
                prevObs = obs
                episode_rewards += reward
            #print 'training episode gained {} rewards in episode {}'.format(episode_rewards, episode)

    def test_agent(self):
        print 'testing TDLearningAgent...'
        episode_rewards = 0
        done, obs = False, self.env.reset()
        while done == False:
            action = self.getPolicy(obs)
            obs, reward, done, _ = self.env.step(action)
            episode_rewards += reward
        print 'testing episode gained {} rewards'.format(episode_rewards)

"""
TODO
"""
class MonteCarloAgent(Agent):
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
    pass
    # for episode in range(self.episodes):
    #     self.observation = self.env.reset()
    #     done = False
    #     total_reward, reward = 0,0

    #     action = numpy.argmax(self.Q[self.observation])

    #     observation, reward, done, info = self.env.step(self.env.action_space.sample())
    #     Q[self.observation, action] += alpha * (reward + gamma * (numpy.max(Q[observation] - Q[self.observation, action])))
    #     self.observation = observation
    #     total_reward += reward


"""
QLearning agent (RL2 last thing scott talked about)
"""
class DeepQLearningAgent(Agent):
    def __init__(self, game_name, episodes):

        self.env = gym.make(game_name)
        self.episodes = episodes
        self.observation = self.env.reset()
        # number 7056 comes from image which is a 84x84x1 color image
        self.scalarInput = tensorflow.placeholder(shape=[None, 7056], dtype=tf.float32)
        self.imageIn = tensorflow.reshpae(self.scalarInput, shape=[-1,84,84,2])

        # create 4 layers of the convolutional neural network
        conv1 = slim.conv2d( \
            inputs=imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', biases_initializer=None)
        conv2 = slim.conv2d( \
            inputs=conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
        conv3 = slim.conv2d( \
            inputs=conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
        conv4 = slim.conv2d( \
            inputs=conv3,num_outputs=512,kernel_size=[7,7],stride=[1,1],padding='VALID', biases_initializer=None)

        # Take the output from the final convolutional layer and split it into separate advantage and value streams.
        streamAC,streamVC = tensorflow.split(conv4,2,3)
        streamA = slim.flatten(streamAC)
        streamV = slim.flatten(streamVC)
        xavier_init = tensorflow.contrib.layers.xavier_initializer()
        AW = tensorflow.Variable(xavier_init([h_size//2,env.actions]))
        VW = tensorflow.Variable(xavier_init([h_size//2,1]))
        Advantage = tensorflow.matmul(streamA,AW)
        Value = tensorflow.matmul(streamV,VW)

        # Combine them together to get our final Q-values
        self.Qout = self.Value + tensorflow.subtract(self.Advantage,tensorflow.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        self.predict = tensorflow.argmax(self.Qout,1)

        # We obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tensorflow.placeholder(shape=[None],dtype=tensorflow.float32)
        self.actions = tensorflow.placeholder(shape=[None],dtype=tensorflow.int32)
        self.actions_onehot = tensorflow.one_hot(self.actions,env.actions,dtype=tensorflow.float32)

        self.Q = tensorflow.reduce_sum(tensorflow.multiply(self.Qout, self.actions_onehot), axis=1)

        self.loss = tensorflow.reduce_mean(tensorflow.square(self.targetQ - self.Q))
        self.trainer = tensorflow.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

    def train_agent(self):
        init = tf.initialize_all_variables()

