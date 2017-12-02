class FrozenLakeMDP(object):
  def __init__(self):
    self.startState = (1, 1)

  def getPos(self, i):
    return ((i % 4) + 1, (i / 4) + 1)

  def getStates(self):
    return [self.getPos(i) for i in range(16)]

  def getActions(self):
    return [i for i in range(4)]

  def isTerminal(self, state):
    return (state == (2,2) or state == (4,2) or state == (4,3) or state == (1,4))

  def isGoal(self, state):
    return state == (4,4)

  def getReward(self, state, action, nextState):
    if self.isGoal(nextState):
      return 1.0
    else:
      return 0.0

  def getTransitionStatesAndProbs(self, state, action):
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

    if state == (1,1):
      return options([(state, tThird), (down, third)], \
                     [(down, third), (right, third), (state, third)], \
                     [(right, third), (down, third), (state, third)], \
                     [(state, tThird), (right, third)], action)

    if state == (1,4):
      return options([(state, tThird), (up, third)], \
                     [(state, tThird), (right, third)], \
                     [(right, third), (up, third), (state, third)], \
                     [(state, third), (right, third), (up, third)], action)

    if state == (4,1):
      return options([(state, third), (down, third), (left, third)], \
                     [(state, third), (left, third), ( down ,third)], \
                     [(state, tThird), (down, third)], \
                     [(state, tThird), (left, third)], action)

    if state == (4,4):
      return options([(state, third), (left, third), (up, third)], \
                     [(state, tThird), (left, third)], \
                     [(state, tThird), (up, third)], \
                     [(state, third), (up, third), (left, third)], action)

    if state == (1,2) or state == (1,3):
      return options([(state, third), (up, third), (down, third)], \
                     [(state, third), (down, third), (right, third)], \
                     [(right, third), (up, third), (down, third)], \
                     [(state, third), (up, third), (right, third)], action)

    if state == (2,1) or state == (3,1):
      return options([(state, third), (left, third), (down, third)], \
                     [(left, third), (down, third), (right, third)], \
                     [(state, third), (right, third), (down, third)], \
                     [(state, third), (left, third), (right, third)], action)

    if state == (4,2) or state == (4,3):
      return options([(left, third), (up, third), (down, third)], \
                     [(state, third), (down, third), (left, third)], \
                     [(state, third), (up, third), (down, third)], \
                     [(state, third), (up, third), (left, third)], action)

    if state == (2,4) or state == (3,4):
      return options([(state, third), (up, third), (left, third)], \
                     [(state, third), (left, third), (right, third)], \
                     [(state, third), (up, third), (right, third)], \
                     [(up, third), (left, third), (right, third)], action)

    if state == (2,2) or state == (2,3) or state == (3,2) or state == (3,3):
      return options([(left, third), (up, third), (down, third)], \
                     [(down, third), (left, third), (right, third)], \
                     [(right, third), (up, third), (down, third)], \
                     [(up, third), (left, third), (right, third)], action)
