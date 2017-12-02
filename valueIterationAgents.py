


class ValueIterationAgent(object):

    def __init__(self, mdp, discount = 0.9, iterations = 100):

        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        cache = dict() # caches values corresponding to a (state, iteration)

        # calculates value of a state at an iteration level
        def valueIteration(state, iteration=0):

          try:
            return cache[state, iteration]

          except: # value has not already been computed
            actions = self.mdp.getPossibleActions(state)
            if actions == [] or self.mdp.isTerminal(state) or iteration >= self.iterations:
              return 0

            maxValue = None
            for action in actions:

              summation = 0
              for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                summation += prob * (self.mdp.getReward(state, action, nextState) + \
                  self.discount * valueIteration(nextState, iteration + 1))

              if maxValue < summation:
                maxValue = summation

            cache[state, iteration] = maxValue # update cache for current value
            return maxValue

        for state in self.mdp.getStates():
          self.values[state] = valueIteration(state)

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        summation = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
          summation += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.getValue(nextState))
        return summation

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
          return None

        maxValue, maxAction = None, None
        for action in self.mdp.getPossibleActions(state):

          QValue = self.computeQValueFromValues(state, action)
          if maxValue is None or maxValue < QValue:
            maxValue = QValue
            maxAction = action

        return maxAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
