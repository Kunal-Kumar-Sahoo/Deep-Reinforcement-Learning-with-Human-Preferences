import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

import warnings; warnings.filterwarnings('ignore')

class CartPoleEnv():
    def __init__(self, mode='human', seed=None):
        self.env = gym.make('CartPole-v1', render_mode=mode)
        if seed is not None:
            self.env.seed(seed)
            self.env.action_space.seed(seed)
        self.state, _ = self.env.reset()
        self.step_count = 0

    def reset(self):
        self.state, _ = self.env.reset()
        self.step_count = 0

    def step(self, action):
        self.step_count += 1
        state, reward, terminated, truncated, _ = self.env.step(action)
        self.state = state
        if terminated or truncated:
            self.reset()
            reward = int(rewarder(torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)).item())
            return reward, False
        return reward, True

    def render(self):
        self.env.render()

class Actor(nn.Module):
    def __init__(self, in_size, num_actions, num_hidden_units):
        super(Actor, self).__init__()
        self.shared_1 = nn.Linear(in_size, num_hidden_units)
        self.actor = nn.Linear(num_hidden_units, num_actions)

    def forward(self, input_obs):
        x = F.relu(self.shared_1(input_obs))
        return self.actor(x)

class Rewarder(nn.Module):
    def __init__(self, in_size, num_hidden_units):
        super(Rewarder, self).__init__()
        self.shared_1 = nn.Linear(in_size, num_hidden_units)
        self.reward = nn.Linear(num_hidden_units, 1)

    def forward(self, input_obs):
        x = F.relu(self.shared_1(input_obs))
        return self.reward(x)

agent = Actor(4, 2, 100)
rewarder = Rewarder(4, 100)
env1 = CartPoleEnv(mode='rgb_array')
env2 = CartPoleEnv(mode='rgb_array')

def calculate_g(reward_trajectory, gamma):
    ez_discount = np.array([gamma ** n for n in range(len(reward_trajectory))])
    gs = []
    reward_trajectory = np.array(reward_trajectory)
    for ts in range(len(reward_trajectory)):
        to_end_rewards = reward_trajectory[ts:]
        eq_len_discount = ez_discount[:len(reward_trajectory[ts:])]
        total_value = np.multiply(to_end_rewards, eq_len_discount)
        g = sum(total_value)
        gs.append(g)
    return gs

def step_episode(env, model):
    env.reset()
    action_probs_list = []
    rewards = []
    states = []
    actions = []
    playing = True
    while playing:
        obs = torch.tensor(env.state, dtype=torch.float32).unsqueeze(0)
        action_logits = agent(obs)
        action_probs = F.softmax(action_logits, dim=-1)
        selected_action_idx = torch.multinomial(action_probs, 1).item()
        states.append(obs)
        actions.append(selected_action_idx)

        reward, playing = env.step(selected_action_idx)

        probability_of_taking_selected_action = action_probs[0, selected_action_idx]
        action_probs_list.append(probability_of_taking_selected_action)
        rewards.append(reward)
    
    return action_probs_list, rewards, states, actions

def actor_loss(action_probs, rewards):
    gs = calculate_g(rewards, 0.99)
    action_log_probs = torch.log(torch.stack(action_probs))
    loss = -torch.sum(action_log_probs * torch.tensor(gs, dtype=torch.float32))
    return loss

optimizer_rewarder = optim.Adam(rewarder.parameters(), lr=0.0005)
optimizer_actor = optim.Adam(agent.parameters(), lr=0.0005)

def decode_action(action):
    actions = {
        0: 'left',
        1: 'right'
    }
    return actions[action]

def render_envs(env1, env2, states1, states2, actions1, actions2):
    frames1 = []
    frames2 = []
    
    for state1, state2 in zip(states1, states2):
        env1.env.state = state1.squeeze().numpy()
        env2.env.state = state2.squeeze().numpy()
        frames1.append(env1.render())
        frames2.append(env2.render())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(frames1[-1])
    axes[0].set_title(f"Action: {decode_action(actions1[-1])}")
    axes[1].imshow(frames2[-1])
    axes[1].set_title(f"Action: {decode_action(actions2[-1])}")
    plt.show()

def preference_update(states1, actions1, rewards1, states2, actions2, rewards2, rewarder):
    # Choose random trajectory indices for comparison
    transition_ids1 = random.sample(range(len(states1)-2), 1)[0]
    transition_ids2 = random.sample(range(len(states2)-2), 1)[0]

    # Render the trajectories for comparison
    render_envs(env1, env2, states1[transition_ids1:], states2[transition_ids2:], actions1[transition_ids1:], actions2[transition_ids2:])
    
    # Ask for user input on preference
    pref = input('select preference a: left, d: right, s: same  ')
    dists = {'a': [1, 0], 'd': [0, 1], 's': [0.5, 0.5]}
    dist = dists[pref]

    # Compute rewards using the rewarder model for the selected states
    reward1 = rewarder(states1[transition_ids1+1])
    reward2 = rewarder(states2[transition_ids2+1])
    
    # Use the cumulative rewards to compute the probabilities
    cum_reward1 = sum(rewards1[:transition_ids1+1])
    cum_reward2 = sum(rewards2[:transition_ids2+1])
    p1 = torch.exp(cum_reward1 + reward1) / (torch.exp(cum_reward1 + reward1) + torch.exp(cum_reward2 + reward2))
    p2 = torch.exp(cum_reward2 + reward2) / (torch.exp(cum_reward1 + reward1) + torch.exp(cum_reward2 + reward2))
    
    # Compute the loss
    loss = -torch.log(p1) * dist[0] - torch.log(p2) * dist[1]
    
    # Update the rewarder model
    optimizer_rewarder.zero_grad()
    loss.backward()
    optimizer_rewarder.step()

average_length = []
for episode in range(5000):
    action_probs1, rewards1, states1, actions1 = step_episode(env1, agent)
    action_probs2, rewards2, states2, actions2 = step_episode(env2, agent)
    
    loss = actor_loss(action_probs1, rewards1)
    
    optimizer_actor.zero_grad()
    loss.backward()
    optimizer_actor.step()
    
    average_length.append(len(rewards1))
    
    preference_update(states1, actions1, rewards1, states2, actions2, rewards2, rewarder)  # Pass rewards here
    
    print('Episode:', episode, '\nAverage steps to target:', np.mean(average_length[-100:]))
