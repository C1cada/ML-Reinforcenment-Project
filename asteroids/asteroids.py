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

def preprocess(I):
    """make the data returned by gym more usable"""
    I = I[35:195] # cut off some of the sides to make it square
    I = I[::2,::2,0] # compact by skipping every other and using shorthand for color
    I[I == 112] = 0 #ignore the score letters
    I[I != 0] = 1 #set everything other than the score and backround to 1
    return I.astype(np.float).ravel() # return the 2d list as a 1d list of floats

env = gym.make("ALE/Asteroids-v5",frameskip=3, render_mode='human',)
observation = env.reset()
prev_state = None
D = 80*80
episode_number = 0
    

while True:
    action = 0
    observation, reward, done, info = env.step(action)
    observation2, reward2, done, info = env.step(action) #need to steps due to the alternating states that are returned
    reward = reward + reward2
    state = preprocess(observation + observation2)
    difference = state - prev_state if prev_state is not None else np.zeros(D)
    prev_state = state

    """
    Check the policy
    Record memory
    """
    if done:
        episode_number += 1
        """
        Adjust policy
        Discount reward
        """

    

    
    # print(observation.shape)
    # observation = observation[35:195]
    # ob = observation[::2,::2, 0]
    # observation[observation == 144] = 0 #turn bacround colors to 0
    # ob[ob == [214, 214, 214]] = 0
    # ob[ob == [184, 50, 50]] = 0
    # observation[observation == ] = 0 #turn backround colors to 0
    # observation[observation != 0] = 1
    # print(ob[ob != 0])
    # print(info)
    # print(len(ob[ob != 0]))