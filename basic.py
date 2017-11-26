import gym
import universe

"""
Maximize your score in the Atari 2600 game SpaceInvaders. In this environment,
the observation is an RGB image of the screen, which is an array of shape (210, 160, 3).
Each action is repeatedly performed for a duration of k frames, where k is
uniformly sampled from {2,3,4}.
"""

env = gym.make('gym-core.SpaceInvaders-v0')
env.configure(remotes=1)
observation_n = env.reset()

print "\nSTART OBS is {}\n".format(observation_n)

while True:
    action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]
    observation_n, reward_n, done_n, info = env.step(action_n)
    print "\nOBS is {}\n".format(observation_n)
env.render()
