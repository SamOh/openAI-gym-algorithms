import util
import random

class TDLearningAgent(object):
    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.99):
      self.QValues = util.Counter()
      self.alpha = alpha
      self.epsilon = epsilon
      self.gamma = gamma

    def getLegalActions(self, state):
        return ["Left", "Right", "Up"]

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.QValues[state, action]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        maxQValue = None
        for action in self.getLegalActions(state):
          QValue = self.getQValue(state, action)
          if maxQValue < QValue:
            maxQValue = QValue

        return 0 if maxQValue is None else maxQValue

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        maxQValue, maxAction = None, None
        for action in self.getLegalActions(state):
          QValue = self.getQValue(state, action)
          if maxQValue < QValue:
            maxQValue = QValue
            maxAction = action
        return maxAction

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        if legalActions == []:
          return None
        else:
          return random.choice(legalActions) if util.flipCoin(self.epsilon) else self.computeActionFromQValues(state)

    def update(self, state, action, reward, nextState):
        self.QValues[state, action] = (1 - self.alpha) * self.QValues[state, action] + \
          self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
