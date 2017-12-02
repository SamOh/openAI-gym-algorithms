"""
An MDP model designed for the Open AI gym game, FrozenLake. The game is
a 4 x 4 grid of ice and holes, which we model with coordinates: (1, 1) at the
top left (the start state), and (4, 4) at the bottom right.
"""

class FrozenLakeMDP(object):
  def __init__(self, maxValue, holes):
    self.max = maxValue
    self.holes = holes

  """
  converts a grid number to an (x, y) coordinate
  """
  def getPos(self, i):
    return ((i % self.max) + 1, (i / self.max) + 1)

  """
  list of all coordinate points of 4x4 grid
  """
  def getStates(self):
    return [self.getPos(i) for i in range(16)]

  """
  FrozenLake has 4 actions: left (0), down (1), right (2), up (3)
  """
  def getActions(self):
    return [i for i in range(4)]

  """
  Check to see if a state is a hole
  """
  def isTerminal(self, state):
    return state in self.holes

  """
  Goalstate is at bottom right
  """
  def isGoal(self, state):
    return state == (self.max, self.max)

  """
  This reward system is consistent with the one specified by open AI
  (1 point for reaching the goal, otherwise 0 points)
  """
  def getReward(self, nextState):
    if self.isGoal(nextState):
      return 1.0
    else:
      return 0.0

  """
  Gives list of (transition states, probability) from another state, only
  allowing legal actions
  """
  def getTransitionStatesAndProbs(self, state, action):

    # helper for choosing which [(trans states, probabilities)] given action
    def options(lst1, lst2, lst3, lst4, action):
      if action == 0:
        return lst1
      if action == 1:
        return lst2
      if action == 2:
        return lst3
      if action == 3:
        return lst4

    third, tThird = 1.0/3, 2.0/3
    x, y = state
    left, down, right, up = (x-1,y), (x,y+1), (x+1,y),(x,y-1)

    # TOP LEFT
    if state == (1, 1):
      return options([(state, tThird), (down, third)], \
                     [(down, third), (right, third), (state, third)], \
                     [(right, third), (down, third), (state, third)], \
                     [(state, tThird), (right, third)], action)

    # BOTTOM LEFT
    elif state == (1, self.max):
      return options([(state, tThird), (up, third)], \
                     [(state, tThird), (right, third)], \
                     [(right, third), (up, third), (state, third)], \
                     [(state, third), (right, third), (up, third)], action)

    # TOP RIGHT
    elif state == (self.max, 1):
      return options([(state, third), (down, third), (left, third)], \
                     [(state, third), (left, third), ( down ,third)], \
                     [(state, tThird), (down, third)], \
                     [(state, tThird), (left, third)], action)

    # BOTTOM RIGHT
    elif state == (self.max, self.max):
      return options([(state, third), (left, third), (up, third)], \
                     [(state, tThird), (left, third)], \
                     [(state, tThird), (up, third)], \
                     [(state, third), (up, third), (left, third)], action)

    # LEFT EDGE
    elif x == 1:
      return options([(state, third), (up, third), (down, third)], \
                     [(state, third), (down, third), (right, third)], \
                     [(right, third), (up, third), (down, third)], \
                     [(state, third), (up, third), (right, third)], action)

    # TOP EDGE
    elif y == 1:
      return options([(state, third), (left, third), (down, third)], \
                     [(left, third), (down, third), (right, third)], \
                     [(state, third), (right, third), (down, third)], \
                     [(state, third), (left, third), (right, third)], action)

    # RIGHT EDGE
    elif x == self.max:
      return options([(left, third), (up, third), (down, third)], \
                     [(state, third), (down, third), (left, third)], \
                     [(state, third), (up, third), (down, third)], \
                     [(state, third), (up, third), (left, third)], action)

    # BOTTOM EDGE
    elif y == self.max:
      return options([(state, third), (up, third), (left, third)], \
                     [(state, third), (left, third), (right, third)], \
                     [(state, third), (up, third), (right, third)], \
                     [(up, third), (left, third), (right, third)], action)

    # MIDDLE
    else:
      return options([(left, third), (up, third), (down, third)], \
                     [(down, third), (left, third), (right, third)], \
                     [(right, third), (up, third), (down, third)], \
                     [(up, third), (left, third), (right, third)], action)
