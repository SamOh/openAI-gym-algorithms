import gym
import universe
import agent
from train import train
from test import test

if __name__ == "__main__":
  env = gym.make('gym-core.SpaceInvaders-v0')
  env.configure(remotes=1)
  player = agent.TDLearningAgent()
  train(player, env)
  test(player, env)

