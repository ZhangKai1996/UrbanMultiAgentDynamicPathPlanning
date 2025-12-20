import os.path
import time
import shutil

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from algo.memory import ScalableReplayBuffer
from algo.misc import soft_update, LinearSchedule
from common.geo import load_graph
from common.utils import haversine

max_step = 100


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.embedding = nn.Embedding(state_dim, 256)

        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x1 = self.embedding(state[:, 0])
        x2 = self.embedding(state[:, 1])
        x = torch.cat([x1, x2], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PathPlannerDQN:
    def __init__(self, graph, p,
                 num_action,
                 tau=1e-3,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 batch_size=32,
                 epsilon_decay=int(1e6),
                 buffer_size=int(1e6),
                 log_folder='trained/logs_pp_dqn',
                 has_writer=True,
                 **kwargs):
        self.p = p
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.num_action = num_action
        self.batch_size = batch_size

        self.cur_node_idx = None
        self.start_node = None
        self.end_node, self.end_node_idx = None, None

        self.state_dim = len(self.nodes)
        self.action_dim = num_action

        self.q_network = DQN(self.state_dim, self.action_dim)
        self.target_network = DQN(self.state_dim, self.action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)

        self.tau = tau
        self.gamma = gamma
        self.replay_buffer = ScalableReplayBuffer(capacity=buffer_size)
        self.schedule = LinearSchedule(epsilon_decay, epsilon_end, epsilon_start)

        self.writer = None
        if has_writer:
            if os.path.exists(log_folder):
                shutil.rmtree(log_folder)
                print('Removing all previous files!')
            self.writer = SummaryWriter(log_dir=log_folder)

    def __get_state(self):
        return [self.end_node_idx, self.cur_node_idx]

    def reset(self):
        start_node_idx, self.end_node_idx = np.random.choice(len(self.nodes), 2, replace=False)
        # start_node_idx, self.end_node_idx = 0, 100

        self.cur_node_idx = start_node_idx
        self.start_node = self.nodes[start_node_idx]
        self.end_node = self.nodes[self.end_node_idx]
        return self.__get_state()

    def choose_action(self, state, t=None):
        if t is not None:
            if np.random.uniform() < self.schedule.value(t):
                return np.random.choice(self.num_action)

        state_tensor = torch.tensor(state, dtype=torch.int32).unsqueeze(0)
        q_values = self.q_network(state_tensor).squeeze(0).detach().numpy()
        indexes = np.argwhere(q_values == q_values.max()).squeeze(axis=1)
        return np.random.choice(indexes)

    def step(self, action, test=False):
        current_node = self.nodes[self.cur_node_idx]
        next_nodes = self.p[current_node]
        if action >= len(next_nodes):
            return self.__get_state(), -10.0, False

        next_node = next_nodes[action]
        next_node_idx = self.nodes.index(next_node)
        done = (next_node_idx == self.end_node_idx)
        if not test:
            nodes = self.graph.nodes
            dist = haversine(nodes[next_node], nodes[self.end_node])
            reward = -dist
            reward += int(done) * 1e2
        else:
            reward = -self.graph[current_node][next_node][0]['dynamic_weight']
            reward += int(done) * 1e2

        self.cur_node_idx = next_node_idx
        return self.__get_state(), reward, done

    def update_q_network(self, t):
        if len(self.replay_buffer) < int(1e4): return None
        if t % 100 != 0: return None

        # Sample a batch of experiences from the buffer
        losses = []
        batch_dict = self.replay_buffer.sample(self.batch_size, num_keys=8)
        total_gradients = None
        for key, batch in batch_dict.items():
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors
            states_tensor = torch.tensor(np.array(states), dtype=torch.int32)
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.int64)
            rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32)
            next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.int32)
            dones_tensor = torch.tensor(np.array(dones), dtype=torch.float32)

            # Compute Q values using current Q network
            q_values = self.q_network(states_tensor)
            next_q_values = self.target_network(next_states_tensor)
            next_q_values_max = next_q_values.max(1)[0]
            target_q_values = rewards_tensor + self.gamma * next_q_values_max * (1 - dones_tensor)
            # Get the Q values for the actions taken
            action_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            # Compute loss
            loss = F.mse_loss(action_q_values, target_q_values.detach())

            # Back propagate to compute gradients
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 1.0)
            # Accumulate gradients
            if total_gradients is None:
                total_gradients = {name: param.grad.clone() for name, param in self.q_network.named_parameters()}
            else:
                for name, param in self.q_network.named_parameters():
                    total_gradients[name] += param.grad
            # Store individual task loss for later analysis
            losses.append(loss.item())

        # After all tasks, average the gradients (if you want to average them)
        for name, param in self.q_network.named_parameters():
            param.grad = total_gradients[name] / len(batch_dict)
        # Update the model parameters with the averaged gradients
        self.optimizer.step()

        if t % 100 == 0:
            soft_update(self.target_network, self.q_network, self.tau)
        return np.mean(losses)

    def save_model(self, save_path):
        torch.save(self.q_network.state_dict(), save_path)

    def load_mode(self, load_path):
        state_dict = torch.load(load_path)
        self.q_network.load_state_dict(state_dict)

    def train(self, num_episodes=1000):
        rew_stats, sr_stats = [], []
        step_stats, loss_stats = [], []
        step = 0

        start_time = time.time()
        for episode in range(1, num_episodes + 1):
            done, state = False, self.reset()
            total_reward, episode_step = 0.0, 0
            while not done:
                action = self.choose_action(state, t=step)
                next_state, reward, done, *_ = self.step(action)
                self.replay_buffer.push(state, action, reward, next_state, float(done),
                                        key=self.end_node_idx)
                # Update Q network
                loss = self.update_q_network(t=step)
                if loss is not None:
                    loss_stats.append(loss)

                state = next_state
                total_reward += reward
                episode_step += 1
                step += 1
                if episode_step >= max_step: break

            step_stats.append(episode_step)
            rew_stats.append(total_reward)
            sr_stats.append(int(done))
            if episode % 100 == 0:
                end_time = time.time()
                mean_step = np.mean(step_stats)
                mean_rew = np.mean(rew_stats)
                mean_loss = np.mean(loss_stats)
                sr = np.mean(sr_stats)
                eps = self.schedule.value(step)

                print("Episode: {}".format(episode), end=', ')
                print("Step: {}".format(step), end=', ')
                print("Mean Step: {:>7.2f}".format(mean_step), end=', ')
                print("Mean Rew: {:>7.2f}".format(mean_rew), end=', ')
                print("SR: {:>6.2f}".format(sr), end=', ')
                print("Epsilon: {:>6.3f}".format(eps), end=',')
                print("Time: {:>5.2f}".format(end_time-start_time))

                if self.writer is not None:
                    self.writer.add_scalar('Mean Step', mean_step, episode)
                    self.writer.add_scalar('Mean Rew', mean_rew, episode)
                    self.writer.add_scalar('Mean Loss', mean_loss, episode)
                    self.writer.add_scalar('SR', sr, episode)
                    self.writer.add_scalar('Epsilon', eps, step)

                rew_stats, sr_stats = [], []
                step_stats, loss_stats = [], []
                start_time = end_time
                self.save_model('trained/model_pp_dqn.pth')

    def test(self, num_episodes=1000, name='fig'):
        self.load_mode('trained/model_pp_dqn.pth')

        stats, gap_stats = [], []
        for ep in range(num_episodes):
            done_rl, state = False, self.reset()
            print('\nEpisode: {}, Start: {}, End: {}'.format(ep, self.start_node, self.end_node))

            # Reinforcement Learning
            total_reward, episode_step = 0.0, 0
            path_rl = [self.nodes[self.cur_node_idx], ]
            while not done_rl:
                action = self.choose_action(state)
                next_state, reward, done_rl = self.step(action, test=True)
                print('\t', state, action, next_state, reward, done_rl)
                path_rl.append(self.nodes[self.cur_node_idx])
                total_reward += reward
                state = next_state
                episode_step += 1
                if episode_step >= max_step: break
            print(total_reward, done_rl)
            cost_rl = 1e2-total_reward
            print("\t Optimal path (RL):", round(cost_rl, 2), done_rl, path_rl)

            # Dijkstra
            path_di = nx.shortest_path(self.graph,
                                       source=self.start_node,
                                       target=self.end_node,
                                       weight='dynamic_weight')
            done_di = path_di[0] == self.start_node and path_di[-1] == self.end_node
            cost_di = 0.0
            for i, node1 in enumerate(path_di[:-1]):
                node2 = path_di[i + 1]
                cost_di += self.graph[node1][node2][0]['dynamic_weight']
            print("\t Optimal path (DI):", round(cost_di, 2), done_di, path_di)

            stats.append([int(done_rl), cost_rl, int(done_di), cost_di])
            if done_rl and done_di:
                gap_stats.append((cost_rl-cost_di)/cost_di)

        mean_values = np.array(stats).mean(axis=0)

        plt.Figure()
        plt.hist(gap_stats, bins=20, alpha=0.6, density=True)
        plt.title('SR(rl):{:>6.3f}, SR(di):{:>6.3f}'.format(mean_values[0], mean_values[2]))
        plt.savefig(name+'.png', dpi=300)
        # plt.show()
        plt.close()


def main():
    # np.random.seed(1)

    # Step 1: 获取地图数据（可选择不同的网络类型，比如“drive”, “bike”, “walk”）
    place_name = "南京航空航天大学(将军路校区)"
    graph, p, num_action = load_graph(place_name)

    # Step 2: 使用 DQN 来训练路径规划智能体
    planner = PathPlannerDQN(graph, p, num_action, epsilon_decay=int(2e8))
    planner.train(num_episodes=int(1e7))


if __name__ == '__main__':
    main()
