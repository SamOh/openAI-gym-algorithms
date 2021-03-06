import tensorflow as tf
import os
import multiprocessing
import threading
from time import sleep
import gym
from a3c import *
from worker import *

max_episode_length = 300
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = 7056 # we convert observations to frames of 84 * 84 * 1
a_size = 6 # action set size of Space Invaders
load_model = False
model_path = './model'

# 0 : "NOOP",
# 1 : "FIRE",
# 2 : "UP",
# 3 : "RIGHT",
# 4 : "LEFT"

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Create a directory to save episode playback gifs to
# if not os.path.exists('./frames'):
#     os.makedirs('./frames')

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)

    # Global Network
    master_network = AC_Network(s_size,a_size,'global',None)

    # Set number of workers to number of available CPU threads
    num_workers = multiprocessing.cpu_count()
    workers = []

    # Create worker classes
    for i in range(num_workers):
        env = gym.make('SpaceInvaders-v0')
        workers.append(Worker(env,i,s_size,a_size,trainer,model_path,global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate thread.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
