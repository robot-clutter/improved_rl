import clt_core as clt

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import math
import os
import pickle


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units):
        super(Critic, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(state_dim + action_dim, hidden_units[0]))
        for i in range(1, len(hidden_units)):
            self.hidden_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
            std_v = 1. / math.sqrt(self.hidden_layers[i].weight.size(1))
            self.hidden_layers[i].weight.data.uniform_(-std_v, std_v)
            self.hidden_layers[i].bias.data.uniform_(-std_v, std_v)

        self.out = nn.Linear(hidden_units[-1], 1)
        std_v = 1. / math.sqrt(self.out.weight.size(1))
        self.out.weight.data.uniform_(-std_v, std_v)
        self.out.bias.data.uniform_(-std_v, std_v)

    def forward(self, x, u):
        x = torch.cat([x, u], x.dim() - 1)
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))
        out = self.out(x)
        return out


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units):
        super(Actor, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(state_dim, hidden_units[0]))
        for i in range(1, len(hidden_units)):
            self.hidden_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))
            std_v = 1. / math.sqrt(self.hidden_layers[i].weight.size(1))
            self.hidden_layers[i].weight.data.uniform_(-std_v, std_v)
            self.hidden_layers[i].bias.data.uniform_(-std_v, std_v)

        self.out = nn.Linear(hidden_units[-1], action_dim)
        std_v = 1. / math.sqrt(self.out.weight.size(1))
        self.out.weight.data.uniform_(-std_v, std_v)
        self.out.bias.data.uniform_(-std_v, std_v)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))
        out = torch.tanh(self.out(x))
        return out

    def forward2(self, x):
        for layer in self.hidden_layers:
            x = nn.functional.relu(layer(x))
        out = self.out(x)
        return out


class DDPG(clt.Agent):
    def __init__(self, state_dim, action_dim, params):
        super(DDPG, self).__init__(name='ddpg', params=params)
        self.visual_state_dim = state_dim['visual']
        self.full_state_dim = state_dim['full']
        self.action_dim = action_dim

        torch.manual_seed(0)

        self.actor = Actor(self.visual_state_dim, action_dim, self.params['actor']['hidden_units'])
        self.target_actor = Actor(self.visual_state_dim, action_dim, self.params['actor']['hidden_units'])

        self.critic = Critic(self.full_state_dim, action_dim, self.params['critic']['hidden_units'])
        self.target_critic = Critic(self.full_state_dim, action_dim, self.params['critic']['hidden_units'])

        self.actor_optimizer = optim.Adam(self.actor.parameters(), self.params['actor']['learning_rate'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), self.params['critic']['learning_rate'])

        self.replay_buffer = clt.ReplayBuffer(self.params['replay_buffer_size'])

        if self.params['noise']['name'] == 'OU':
            self.exploration_noise = clt.OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim),
                                                                  sigma=self.params['noise']['sigma'])
        elif self.params['noise']['name'] == 'Normal':
            self.exploration_noise = clt.NormalNoise(mu=np.zeros(self.action_dim), sigma=self.params['noise']['sigma'])
        else:
            raise ValueError('Exploration noise should be OU or Normal.')

        self.learn_step_counter = 0
        self.save_buffer = False

        self.info['critic_loss'] = 0
        self.info['actor_loss'] = 0

    def predict(self, state):
        s = torch.FloatTensor(state['visual'][0]).to(self.params['device'])
        action = self.actor(s).cpu().detach().numpy()
        return action

    def explore(self, state):
        # Calculate epsilon for epsilon-greedy
        start = self.params['epsilon_start']
        end = self.params['epsilon_end']
        decay = self.params['epsilon_decay']
        epsilon = end + (start - end) * math.exp(-1 * self.learn_step_counter / decay)

        if self.rng.uniform(0, 1) >= epsilon:
            print('predict')

            action = self.predict(state) + self.exploration_noise()
            action[action > 1] = 1
            action[action < -1] = -1
            return action
        else:
            print('explore')
            action = self.rng.uniform(-1, 1, self.action_dim)
            return action

    def q_value(self, state, action):
        s = torch.FloatTensor(state['full']).to(self.params['device'])
        a = torch.FloatTensor(action).to(self.params['device'])
        return self.critic(s, a).cpu().detach().numpy()

    def learn(self, transition):
        # Store transition to the replay buffer
        self.replay_buffer.store(transition)

        # If we have not enough samples just keep storing transitions to the
        # buffer and thus exit from learn.
        if self.replay_buffer.size() < self.params['init_replay_buffer_size']:
            return

        if not self.save_buffer:
            self.replay_buffer.save(os.path.join(self.params['log_dir'], 'replay_buffer'))
            self.save_buffer = True
        
        # Sample a batch
        batch = self.replay_buffer.sample_batch(self.params['batch_size'])
        batch.terminal = np.array(batch.terminal.reshape((batch.terminal.shape[0], 1)))
        batch.reward = np.array(batch.reward.reshape((batch.reward.shape[0], 1)))
        # batch.action = np.array(batch.action.reshape((batch.action.shape[0], 1)))

        visual_state = torch.zeros((self.params['batch_size'], self.visual_state_dim)).to(self.params['device'])
        full_state = torch.zeros((self.params['batch_size'], self.full_state_dim)).to(self.params['device'])
        next_visual_state = torch.zeros((self.params['batch_size'], self.visual_state_dim)).to(self.params['device'])
        next_full_state = torch.zeros((self.params['batch_size'], self.full_state_dim)).to(self.params['device'])

        # ToDo: change state['visual'][0]
        for i in range(self.params['batch_size']):
            visual_state[i] = torch.FloatTensor(batch.state[i]['visual'][0])
            full_state[i] = torch.FloatTensor(batch.state[i]['full'])
            next_visual_state[i] = torch.FloatTensor(batch.next_state[i]['visual'][0])
            next_full_state[i] = torch.FloatTensor(batch.next_state[i]['full'])

        action = torch.FloatTensor(batch.action).to(self.params['device'])
        terminal = torch.FloatTensor(batch.terminal).to(self.params['device'])
        reward = torch.FloatTensor(batch.reward).to(self.params['device'])

        # Compute the target Q-value
        target_q = self.target_critic(next_full_state, self.target_actor(next_visual_state))
        target_q = reward + ((1 - terminal) * self.params['discount'] * target_q).detach()

        # Get the current q estimate
        q = self.critic(full_state, action)

        # Critic loss
        critic_loss = nn.functional.mse_loss(q, target_q)
        self.info['critic_loss'] = float(critic_loss.detach().cpu().numpy())

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute preactivation
        state_abs_mean = self.actor.forward2(visual_state).abs().mean()
        preactivation = (state_abs_mean - torch.tensor(1.0)).pow(2)
        if state_abs_mean < torch.tensor(1.0):
            preactivation = torch.tensor(0.0)
        weight = self.params['actor'].get('preactivation_weight', .05)
        preactivation = weight * preactivation

        actor_loss = -self.critic(full_state, self.actor(visual_state)).mean() + preactivation

        self.info['actor_loss'] = float(actor_loss.detach().cpu().numpy())

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.params['tau'] * param.data + (1 - self.params['tau']) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.params['tau'] * param.data + (1 - self.params['tau']) * target_param.data)

        self.learn_step_counter += 1

    def seed(self, seed=None):
        super(DDPG, self).seed(seed)
        self.replay_buffer.seed(seed)

    def save(self, save_dir, name):
        # Create directory
        log_dir = os.path.join(save_dir, name)
        os.makedirs(log_dir)

        # Save networks and log data
        torch.save({'actor': self.actor.state_dict(),
                    'target_actor': self.target_actor.state_dict()}, os.path.join(log_dir, 'actor.pt'))
        torch.save({'critic': self.critic.state_dict(),
                    'target_critic': self.target_critic.state_dict()}, os.path.join(log_dir, 'critic.pt'))
        log_data = {'params': self.params.copy(),
                    'learn_step_counter': self.learn_step_counter,
                    'state_dim': {'visual': self.visual_state_dim, 'full': self.full_state_dim},
                    'action_dim': self.action_dim}
        pickle.dump(log_data, open(os.path.join(log_dir, 'log_data.pkl'), 'wb'))

    @classmethod
    def load(cls, log_dir):
        log_data = pickle.load(open(os.path.join(log_dir, 'log_data.pkl'), 'rb'))
        self = cls(state_dim=log_data['state_dim'],
                   action_dim=log_data['action_dim'],
                   params=log_data['params'])

        self.loaded = log_dir

        checkpoint_actor = torch.load(os.path.join(log_dir, 'actor.pt'))
        self.actor.load_state_dict(checkpoint_actor['actor'])
        self.target_actor.load_state_dict(checkpoint_actor['target_actor'])

        checkpoint_critic = torch.load(os.path.join(log_dir, 'critic.pt'))
        self.critic.load_state_dict(checkpoint_critic['critic'])
        self.target_critic.load_state_dict(checkpoint_critic['target_critic'])

        return self
