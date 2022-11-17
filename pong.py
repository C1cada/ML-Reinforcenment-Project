import numpy as np
import gym
import pickle
# from gym.utils.play import play
# play(gym.make('Pong-v0'))
# env = gym.make("ALE/Pong-v5", render_mode="human")
H = 200 #hidden layer nodes
batch_size = 10 #episodes per param update?
learning_rate = 0.01
gamma = 0.99 #discount factor for reward
decay_rate = 0.99 #decay factor for RMSProp???
resume = False #resume from a checkpoint
render = False

# model init
D = 80*80 # grid dimensions
if resume:
    model = pickle.load(open('save.p', 'rb')) #loads the model from a previous save
else:
    model = {} #stores the neural net
    model['W1'] = np.random.randn(H,D)/np.sqrt(D) # Xavier init???? normal distrobution
    model['W2'] = np.random.randn(H)/np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v, in model.items()} #buffers that add up gradients over a batch???
rmsprop_cache = { k : np.zeros_like(v) for k,v, in model.items()} # memory for rmsprop

def sigmoid(x):
    return 1.0/ (1.0 + np.exp(-x)) # sigmoid function

def prepro(I):
    # turn 210x160x3 uint8 frame into 80x80 1d float vector
    I = I[35:195]
    I = I[::2,::2,0]
    I[I == 144] = 0 #turn bacround colors to 0
    I[I == 109] = 0 #turn backround colors to 0
    I[I != 0] = 1 #turn everything else to 1
    return I.astype(np.float).ravel() # set all to float and order??

def discount_rewards(r):
    #take 1d float array of rewards and compute discounted reward
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)): #needs a thing for size??
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)??? what does this mean
        running_add = running_add*gamma + r[t] # continuouslt add rewards but not 1 to 1
        discounted_r[t] = running_add # set the discounted reward
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0 #ReLU nonlinearity??
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h #return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
    #backward pass? (eph is array of intermediate hidden states)?
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2']) # vector mutliplication
    dh[eph <= 0] = 0 #backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, "W2":dW2}

if render:
    env = gym.make('Pong-v0', render_mode='human')
else:
    env = gym.make('Pong-v0')
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
while True:
    if render: 
        env.render('rgb_array')

    #preprocess the observation, set input to the difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3

    # record various intermediates (needed later for backprop)
    xs.append(x) #observation
    hs.append(h) #hidden state
    y = 1 if action == 2 else 0
    dlogps.append(y - aprob) #gradient that encourages the action that was taken to be taken

    #step the environment
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward) # record reward to get reward for prev action

    if done: # episode is finished
        episode_number += 1

        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory

        #compute discounted reward
        discounted_epr = discount_rewards(epr)
        #standardize the rewards to be unit normal(helps control gradient estimator)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        
        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.) ????
        grad = policy_backward(eph, epdlogp)
        for k in model:grad_buffer[k] += grad[k] #accumulate grad over patch

        # perform rmsprop parameter epdate every batch_size episodes
        if episode_number % batch_size == 0:
            for k,v in model.items():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1-decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k])+1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        if episode_number % 100 == 0: pickle.dump(model, open("save.p", "wb"))
        reward_sum = 0
        observation = env.reset()
        prev_x = None
    

    if reward != 0.000000: #pong has either +1 or -1 reward exactly whe game ends.
        print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))

   
