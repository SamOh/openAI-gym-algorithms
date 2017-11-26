import gym
import universe

"""
where we test our agent
"""

def test(agent, env):
  state = env.reset()
  total_rewards = 0.0
  while True:
    action = agent.getPolicy(currentState)
    state, reward, done, info = env.step(util.getAtariAction(action))
    total_rewards += reward
    if done:
      break
  print "total rewards were {}".format(total_rewards)
