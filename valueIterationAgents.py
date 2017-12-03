import gym
import utils as util

class ValueIterationAgent(object):

    def __init__(self, mdp, discount, iterations):

        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        cache = {} # caches values corresponding to a (state, iteration)

        # calculates value of a state at an iteration level
        def valueIteration(state, iteration=0):

          try:
            return cache[state, iteration]

          except: # value has not already been computed
            actions = self.mdp.getActions()
            if actions == [] or self.mdp.isTerminal(state) or iteration >= self.iterations:
              return 0

            maxValue = None
            for action in actions:

              summation = 0
              for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                summation += prob * (self.mdp.getReward(nextState) + \
                  self.discount * valueIteration(nextState, iteration + 1))

              if maxValue < summation:
                maxValue = summation

            cache[state, iteration] = maxValue # update cache for current value
            return maxValue

        for state in self.mdp.getStates():
          self.values[state] = valueIteration(state)

    def getValue(self, state):
        return self.values[state]

    def getQValue(self, state, action):
        summation = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
          summation += prob * (self.mdp.getReward(nextState) + self.discount * self.getValue(nextState))
        return summation

    def getPolicy(self, state):
        if self.mdp.isTerminal(state):
          return None

        maxValue, maxAction = None, None
        for action in self.mdp.getActions():

          QValue = self.getQValue(state, action)
          if maxValue is None or maxValue < QValue:
            maxValue = QValue
            maxAction = action

        return maxAction

    def test_agent(self, game, episodes):
      env = gym.make(game)
      env.reset()
      rewards = 0
      for _ in range(episodes):
        obs, done = env.reset(), False
        while not done:
          action = self.getPolicy(self.mdp.getPos(obs))
          obs, reward, done, _ = env.step(action)
          rewards += reward
      print 'Percent success rate was {}%'.format(rewards*100.0/episodes)
