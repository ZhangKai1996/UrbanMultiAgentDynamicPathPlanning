from copy import deepcopy

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Adam

from algo.memory import ScalableReplayBuffer
from algo.misc import soft_update, LinearSchedule
from algo.model import DuelingDQN

from memory import ReplayMemory, Experience
from model import Critic, Actor


class Trainer:
    def __init__(self, n_agents, graph, p,
                 tau=1e-2,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 batch_size=32,
                 epsilon_decay=int(1e6),
                 buffer_size=int(1e6),
                 emb_dim=32,
                 seed=1234):
        th.manual_seed(seed)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = n_agents
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.memory = ReplayMemory(buffer_size)
        self.schedule = LinearSchedule(epsilon_decay, epsilon_end, epsilon_start)

        self.actors, self.critics = [], []
        self.actors_target, self.critics_target = [], []
        self.actors_optimizer, self.critics_optimizer = [], []
        for _ in range(n_agents):
            actor = Actor(dim_obs, n_agents)
            self.actors.append(actor)
            self.actors_target.append(deepcopy(actor))
            self.actors_optimizer.append(Adam(actor.parameters(), lr=1e-4))

            critic = Critic(n_agents, dim_obs, dim_act)
            self.critics.append(critic)
            self.critics_target.append(deepcopy(critic))
            self.critics_optimizer.append(Adam(critic.parameters(), lr=1e-4))
        self.mse_loss = nn.MSELoss()

    def epsilon(self, t): return self.schedule.value(t)

    def choose_action(self, states, episode=None):
        if episode is not None:
            if np.random.uniform() < self.epsilon(episode):
                actions = [np.random.choice(self.n_agents) for _ in states]
                return np.array(actions, dtype=np.int32)

        actions = th.zeros(self.n_agents, self.n_actions)
        for i in range(self.n_agents):
            state = states[i, :].detach().unsqueeze(0)
            act = self.actors[i](state).squeeze()
            actions[i, :] = act
        return actions.data.cpu()

    def add_experience(self, obs_n, act_n, next_obs_n, rew_n, done_n):
        self.memory.push(obs_n, act_n, next_obs_n, rew_n, done_n)

    def update(self, step, update_rate=1):
        if step < int(1e4) or step % update_rate != 0: return None

        c_loss, a_loss = [], []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))

            state_batch = th.from_numpy(np.array(batch.states))
            action_batch = th.from_numpy(np.array(batch.actions))
            next_states_batch = th.from_numpy(np.array(batch.next_states))
            reward_batch = th.from_numpy(np.array(batch.rewards)).unsqueeze(dim=-1)
            done_batch = th.from_numpy(np.array(batch.dones)).unsqueeze(dim=-1)

            self.actors_optimizer[agent].zero_grad()
            current_q = self.critics[agent](state_batch, action_batch)
            next_actions = th.stack([self.actors_target[i](next_states_batch[:, i])
                                     for i in range(self.n_agents)], dim=-1)
            target_next_q = self.critics_target[agent](next_states_batch, next_actions)
            target_q = target_next_q * self.gamma * (1 - done_batch[:, agent, :]) + reward_batch[:, agent, :]
            loss_q = self.mse_loss(current_q, target_q.detach())
            loss_q.backward()
            self.critics_optimizer[agent].step()

            self.actors_optimizer[agent].zero_grad()
            ac = action_batch.clone()
            ac[:, agent, :] = self.actors[agent](state_batch[:, agent, :])
            loss_p = -self.critics[agent](state_batch, ac).mean()
            loss_p.backward()
            self.actors_optimizer[agent].step()

            c_loss.append(loss_q)
            a_loss.append(loss_p)

        for i in range(self.n_agents):
            soft_update(self.critics_target[i], self.critics[i], self.tau)
            soft_update(self.actors_target[i], self.actors[i], self.tau)
        return c_loss, a_loss

    def load_model(self, load_path=None):
        if load_path is None: raise NotImplementedError

        for i, (a, c) in enumerate(zip(self.actors, self.critics)):
            a_state_dict = th.load(load_path + 'actor_{}.pth'.format(i)).state_dict()
            c_state_dict = th.load(load_path + 'critic_{}.pth'.format(i)).state_dict()
            a.load_state_dict(a_state_dict)
            c.load_state_dict(c_state_dict)
            self.actors_target[i] = deepcopy(a)
            self.critics_target[i] = deepcopy(c)

    def save_model(self, save_path=None):
        if save_path is None: raise NotImplementedError

        for i, (a, c) in enumerate(zip(self.actors, self.critics)):
            th.save(a, save_path + 'actor_{}.pth'.format(i))
            th.save(c, save_path + 'critic_{}.pth'.format(i))


class PPDQNTrainer:
    def __init__(self, graph, p,
                 num_action,
                 lr=1e-3,
                 tau=1e-2,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 batch_size=32,
                 epsilon_decay=int(1e6),
                 buffer_size=int(1e6),
                 emb_dim=32,
                 seed=1234):
        th.manual_seed(seed)

        self.p = p
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.num_action = num_action
        self.batch_size = batch_size
        self.state_dim = len(self.nodes)

        print('------------Network kwargs-----------')
        print('\t State dim:', self.state_dim)
        print('\t Action dim:', num_action)
        print('\t Embedding dim:', emb_dim)
        print('\t Learning rate:', lr)

        self.q_network = DuelingDQN(self.state_dim, num_action, emb_dim=emb_dim)
        self.target_network = DuelingDQN(self.state_dim, num_action, emb_dim=emb_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.tau = tau
        self.gamma = gamma
        self.replay_buffer = ScalableReplayBuffer(capacity=buffer_size)
        self.schedule = LinearSchedule(epsilon_decay, epsilon_end, epsilon_start)

    def epsilon(self, t):
        return self.schedule.value(t)

    def choose_action(self, state, mask=None, episode=None):
        if episode is not None:
            if np.random.uniform() < self.epsilon(episode):
                num_action = len(np.where(mask != -np.inf)[0])
                return np.random.choice(num_action)

        state_tensor = th.tensor(state, dtype=th.float32).unsqueeze(0)
        q_values = self.q_network(state_tensor)[0].detach().numpy()
        q_values += mask
        indexes = np.argwhere(q_values == q_values.max()).squeeze(axis=1)
        return np.random.choice(indexes)

    def compute_v(self):
        """
        Compute state value function V(s) = max_a Q(s,a)
        """
        vs = []
        for u_idx, u in enumerate(self.nodes):
            state_tensor = th.tensor([u_idx, ], dtype=th.float32).unsqueeze(0)
            q_values = self.q_network(state_tensor)[0].detach().numpy()
            mask = np.zeros((self.num_action, ))
            mask[len(self.p[u]):] = -np.inf
            vs.append(np.max(q_values + mask))
        return np.array(vs)

    def update_q_network(self, t, update_rate=10):
        if t < int(1e4) or t % update_rate != 0: return None

        # Sample a batch of experiences from the buffer
        batch = self.replay_buffer.sample_(self.batch_size, num_keys=1)
        if len(batch) <= 0: return None

        states, actions, rewards, next_states, next_masks, dones = zip(*batch)
        # Convert to tensors
        states_tensor = th.tensor(np.array(states), dtype=th.float32)
        actions_tensor = th.tensor(np.array(actions), dtype=th.int64)
        rewards_tensor = th.tensor(np.array(rewards), dtype=th.float32)
        next_states_tensor = th.tensor(np.array(next_states), dtype=th.float32)
        next_masks_tensor = th.tensor(np.array(next_masks), dtype=th.float32)
        dones_tensor = th.tensor(np.array(dones), dtype=th.float32)

        # Compute Q values using current Q network
        q_values = self.q_network(states_tensor)
        # Get the Q values for the actions taken
        action_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        # Double DQN target
        next_q_values_online = self.q_network(next_states_tensor)
        next_actions = (next_q_values_online+next_masks_tensor).argmax(1, keepdim=True)
        next_q_values_target = self.target_network(next_states_tensor)
        next_q_values_selected = next_q_values_target.gather(1, next_actions).squeeze(1)
        target_q_values = rewards_tensor + self.gamma * next_q_values_selected * (1 - dones_tensor)
        # Compute loss
        loss = F.mse_loss(action_q_values, target_q_values.detach())
        # Back propagate to compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent explosion
        # torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 1.0)
        th.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        # Update the model parameters with the gradients
        self.optimizer.step()

        soft_update(self.target_network, self.q_network, self.tau)
        return loss.item()

    def save_model(self, save_path):
        th.save(self.q_network.state_dict(), save_path)

    def load_mode(self, load_path):
        state_dict = th.load(load_path)
        self.q_network.load_state_dict(state_dict)
