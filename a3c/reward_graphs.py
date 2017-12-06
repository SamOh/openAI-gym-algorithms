import matplotlib.pyplot as plt

for i in range(8):
  file = open('./rewards/worker{}.txt'.format(i), 'r')
  rewards = []
  for line in file:
    rewards.append(int(line.rstrip('\n')))
  indices = [i for i in range(1,len(rewards)+1)]
  plt.plot(indices, rewards)
  plt.show()
