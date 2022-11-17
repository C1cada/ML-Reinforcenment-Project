import numpy as np
import gym
import pickle

H = 200 #hidden layer nodes
batch_size = 10 #episodes per param update?
learning_rate = 1e-4
gamma = 0.99 #discount factor for reward
decay_rate = 0.99 #decay factor for RMSProp???
resume = True #resume from a checkpoint
render = True

d = 80*80