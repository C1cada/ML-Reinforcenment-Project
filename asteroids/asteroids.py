"""
Asteroids:
    Action Space: 18
    Observation Space: (210, 160, 3)
    Observation High: 255?
    Observation low: 0?

RL:
    Reward:
        Score?
        Time Alive?

        Small positive for being alive
        Big negative for dying

    Data:
        Difference of last and current frame to see motion
        Preprocess to 105 by 105?



"""

import gym
import numpy as np
env = gym.make("ALE/Asteroids-v5",frameskip=3, render_mode='human',)
observation = env.reset()
print(type(observation))
print(observation.shape)
print(observation[0][0])
print(env.action_space)

def merge_frames():
    pass

def preprocess(I):
    I = I[35:195]
    I = I[::2,::2,0]
    I[I == 112] = 0
    

while True:
    action = 0
    observation, reward, done, info = env.step(action)
    observation2, reward2, done2, info2 = env.step(action)
    # observation = preprocess(observation + observation2)
    observation = observation + observation2
    reward = reward + reward2

    

    
    # print(observation.shape)
    # observation = observation[35:195]
    ob = observation[::2,::2, 0]
    # observation[observation == 144] = 0 #turn bacround colors to 0
    # ob[ob == [214, 214, 214]] = 0
    # ob[ob == [184, 50, 50]] = 0
    # observation[observation == ] = 0 #turn backround colors to 0
    # observation[observation != 0] = 1
    print(ob[ob != 0])
    # print(info)
    # print(len(ob[ob != 0]))