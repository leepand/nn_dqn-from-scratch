import gym
from policy import DQNAgent
import numpy as np
from optimx import make
import network as nn

model_name = f"personalized_unispinux_strategy"
VERSION="0.0"
MODEL_ENV="dev"
model_db2 = make(
            f"cache/{model_name}-v{VERSION}",
            db_name="dqnmodel_test.db",
            env=MODEL_ENV,
    db_type="diskcache"
    
        )

env=gym.make('CartPole-v1')


import numpy as np
from collections import deque

# Global variables
NUM_EPISODES = 10000
MAX_TIMESTEPS = 1000
AVERAGE_REWARD_TO_SOLVE = 1950
NUM_EPS_TO_SOLVE = 100
NUM_RUNS = 20
GAMMA = 0.95
EPSILON_DECAY = 0.997
update_size = 10
hidden_layer_size = 24
num_hidden_layers = 2
action_list = [a for a in range(env.action_space.n)]
model  = DQNAgent(actions=action_list,input_dim=env.observation_space.shape[0]
                         ,hidden_dim=32,model_db=model_db2)
#RLAgent(env,num_hidden_layers,hidden_layer_size,GAMMA,EPSILON_DECAY)
scores_last_timesteps = deque([], NUM_EPS_TO_SOLVE)

model_id="ooo1"
# The main program loop
log_interval=100
avg_reward = []
for i_episode in range(NUM_EPISODES):
    ep_reward = 0
    observation,_ = env.reset()
    if i_episode >= NUM_EPS_TO_SOLVE:
        if (sum(scores_last_timesteps)/NUM_EPS_TO_SOLVE > AVERAGE_REWARD_TO_SOLVE):
            print("solved after {} episodes".format(i_episode))
            break
    # Iterating through time steps within an episode
    for t in range(MAX_TIMESTEPS):
        # env.render()
        state=observation[True,:]
        #print(state,state.shape)
        action = model.act(state,model_id=model_id)
        prev_obs = state
        observation, reward, done, info,_ = env.step(action)
        next_state = observation[True,:]
        ep_reward += reward
        # Keep a store of the agent's experiences
        #model.remember(done, action, observation, prev_obs)
        #model.experience_replay(update_size)
        model.learn(prev_obs, action, next_state, reward, model_id, done=done)
        # epsilon decay
        if done:
            # If the pole has tipped over, end this episode
            scores_last_timesteps.append(t+1)
            break
            
        if i_episode % log_interval == 0 and len(avg_reward)>0:
            print("Ave reward: {}".format(sum(avg_reward) / len(avg_reward)))
            avg_reward = []

        else:
            avg_reward.append(ep_reward)
