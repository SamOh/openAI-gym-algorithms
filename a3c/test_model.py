import tensorflow as tf
import numpy as np
from a3c import *
import gym

model_path = './model'
a_size = 6
s_size = 7056
actions = np.identity(a_size,dtype=bool).tolist()

tf.reset_default_graph()
coord = tf.train.Coordinator()

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)

    # Global Network
    master_network = AC_Network(s_size,a_size,'global',None)

    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
  print ('Loading Model...')
  ckpt = tf.train.get_checkpoint_state(model_path)
  saver.restore(sess,ckpt.model_checkpoint_path)

  rnn_state = master_network.state_init
  env = gym.make('SpaceInvaders-v0')
  s = env.reset()
  s = process_frame(s)
  d, total_rewards, steps, max_reward = False, 0, 0, 0

  while not d:
    a_dist,v,rnn_state = sess.run([master_network.policy,master_network.value,master_network.state_out],
                            feed_dict={master_network.inputs:[s],
                            master_network.state_in[0]:rnn_state[0],
                            master_network.state_in[1]:rnn_state[1]})
    a = np.random.choice(a_dist[0],p=a_dist[0])
    a = np.argmax(a_dist == a)

    env.render()
    s, r, d, _ = env.step(actions[a])
    if r > max_reward:
      max_reward = r
    s = process_frame(s)
    total_rewards += r
    steps += 1

  print 'game ended with {} rewards in {} steps'.format(total_rewards, steps)
  print 'max reward earned was {}'.format(max_reward)


