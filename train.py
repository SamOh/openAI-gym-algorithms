import gym
import universe
import util

"""
Where we train our agent
"""

def train(agent, env, numTraining=100):
  episodes = 0
  startState, currentState = env.reset(), ""
  print "\n\nSTART STATE IS {}\n\n".format(startState)
  for (r, g, b) in startState:
    currentState += str(r) + str(g) + str(b)

  for _ in range(numTraining):
    action = agent.getAction(currentState)
    state, reward, done, info = env.step(util.getAtariAction(action))
    nextState = ""
    for (r, g, b) in nextState:
      currentState += str(r) + str(g) + str(b)
    agent.update(currentState, action, reward, nextState)
    episodes += 1
    if done: break

  return agent
