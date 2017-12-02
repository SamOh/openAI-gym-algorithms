"""
Below is an implementation of the open AI game taxi, where we use A* search
to compute the optimal path from the start state of the game.
"""

import gym
from utils import PriorityQueue

"""
gets (x, y) position from map number
"""
def getPosition(obs):
  hundreds, tens = (obs - obs % 100) / 100, ((obs % 100) - (obs % 10)) / 10
  return (tens + 2) / 2, 5 - hundreds

"""
returns (resultingPoint, actionUsed) for a given taxi position
"""
def getSuccessors(pos):
  x, y = pos
  l, r, u, d = ((x-1,y), 3), ((x+1,y), 2), ((x,y+1), 1), ((x,y-1), 0)

  if x == 1:
    if y == 5:
      return [r, d]
    elif y == 1:
      return [u]
    elif y == 2:
      return [d, u]
    else:
      return [r, d, u]
  elif x == 2:
    if y == 5:
      return [l, d]
    elif y == 1:
      return [r, u]
    elif y == 2:
      return [r, u, d]
    else:
      return [l, r, u, d]
  elif x == 3:
    if y == 5:
      return [r, d]
    elif y == 1:
      return [l, u]
    elif y == 2:
      return [l, u, d]
    else:
      return [l, r, u, d]
  elif x == 4:
    if y == 5:
      return [l, r, d]
    elif y == 1:
      return [r, u]
    elif y == 2:
      return [r, u, d]
    else:
      return [l, r, u, d]
  else:
    if y == 5:
      return [l, d]
    elif y == 1:
      return [l, u]
    else:
      return [l, u, d]

"""
heuristic for A* search
"""
def manhattanDistance(p1, p2):
  return (abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]))

"""
A* search for current pos to goal (returns list of actions)
"""
def aStarSearch(pos, goal):
  fringe, visited, best = PriorityQueue(), set(), {}
  fringe.push((pos, [], 0), manhattanDistance(pos, goal))

  while not fringe.isEmpty():

    current_point, actions, total_cost = fringe.pop()

    if current_point in visited or \
    (current_point in best and best[current_point] <= total_cost):
      continue

    visited.add(current_point)
    best[current_point] = total_cost

    # current vertex is a solution
    if current_point == goal:
      return actions

    for (point, action) in getSuccessors(current_point):
      # if node not visited add it to the fringe
      if point not in visited:
        actions_copy = list(actions)
        actions_copy.append(action)
        cost = total_cost + 1
        fringe.push((point, actions_copy, cost), \
          cost + manhattanDistance(point, goal))

  raise Exception('Problem Not Solved')

"""
computes action sequence starting at X1 and ending at X2
"""
def computeActionSequence(X1, X2, pos):
  pickup = aStarSearch(pos, X1)
  pickup.append(4)
  dropoff = aStarSearch(X1, X2)
  dropoff.append(5)
  return pickup + dropoff

"""
solves taxi game based on initial observation
"""
def solveTaxi(obs):
  tens, ones = ((obs % 100) - (obs % 10)) / 10, obs % 10
  G, B, Y, R, pos = (5, 5), (4, 1), (1, 1), (1, 5), getPosition(obs)

  if (tens % 2) == 1:
    if ones == 1:
      return computeActionSequence(Y, B, pos)
    if ones == 2:
      return computeActionSequence(B, R, pos)
    if ones == 3:
      return computeActionSequence(B, G, pos)
    if ones == 4:
      return computeActionSequence(B, Y, pos)
  else:
    if ones == 1:
      return computeActionSequence(R, G, pos)
    if ones == 2:
      return computeActionSequence(R, Y, pos)
    if ones == 3:
      return computeActionSequence(R, B, pos)
    if ones == 4:
      return computeActionSequence(G, R, pos)
    if ones == 6:
      return computeActionSequence(G, Y, pos)
    if ones == 7:
      return computeActionSequence(G, B, pos)
    if ones == 8:
      return computeActionSequence(Y, R, pos)
    if ones == 9:
      return computeActionSequence(Y, G, pos)

if __name__ == "__main__":
  env = gym.make('Taxi-v2')
  correct, iterations = 0, 1
  print "Checking accuracy with {} iterations...".format(iterations)
  for _ in range(iterations):
    obs = env.reset()
    print obs
    env.render()
    actions = solveTaxi(obs)
    for action in actions:
      obs, _, done, _ = env.step(action)
    if done:
      correct += 1
  print '{}% correct'.format(correct*100.0/iterations)
